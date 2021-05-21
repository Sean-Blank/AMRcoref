import random
import math
import numpy as np
from grn import *
from modules import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, codecs

import utils
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from Graph import GraphEncoder

class AMRCorefModel(nn.Module):
    def __init__(self, args, vocabs):
        super(AMRCorefModel, self).__init__()
        self.vocabs = vocabs
        self.args = args
        self.embed_dim = args.embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.emb_dropout = nn.Dropout(self.args.emb_dropout)
        # amr encoder
        self.concept_encoder = TokenEncoder(vocabs['concept'], vocabs['concept_char'],
                                            args.concept_char_dim, args.concept_dim, args.embed_dim,
                                            args.cnn_filters, args.char2concept_dim, args.emb_dropout)
        self.concept_embed_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.relation_emb = nn.Embedding(self.args.relation)
        self.mention_emb_size = self.embed_dim

        # text encoder and text bert embed
        self.word_encoder = TokenEncoder(vocabs['token'], vocabs['token_char'],
                                         args.word_char_dim, args.word_dim, args.embed_dim,
                                         args.cnn_filters, args.char2word_dim, args.emb_dropout)
        self.token_embed_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.args.use_token:
            self.lstm = nn.LSTM(self.embed_dim, self.args.bilstm_hidden_dim,
                                num_layers=self.args.bilstm_layer_num, bidirectional=True)
            self.mention_emb_size = self.embed_dim + self.args.bilstm_hidden_dim * self.args.bilstm_layer_num
        # self.bert = BertModel(bert_config)
        # self.bert_size = bert_config.hidden_size

        if args.use_bert:
            self.mention_emb_size = self.embed_dim + self.bert_size

        # graph encoder
        self.args.edge_vocab_size = vocabs['relation'].size
        self.graph_encoder = GraphEncoder(self.args)

        # add params for ARG
        if args.use_classifier:
            self.arg_feature_dim = self.args.arg_feature_dim
            # loss

            # self.arg_loss = focal_loss(alpha=0.5, gamma=2, num_classes=5, size_average=True)
            self.arg_loss = nn.CrossEntropyLoss()

            self.arg_emb = Embedding(5, self.arg_feature_dim, 0)
            self.arg_classification_layer = FFNN(self.args.ffnn_depth, self.mention_emb_size, self.args.ff_embed_dim, 5,
                                                 self.args.ffnn_dropout)
            self.mention_emb_size = self.mention_emb_size + self.arg_feature_dim

        # mention_score
        self.mention_score = FFNN(self.args.ffnn_depth, self.mention_emb_size, self.args.ff_embed_dim, 1, self.args.ffnn_dropout)

        # fast score
        self.fast_src_projector = linear(self.mention_emb_size, self.mention_emb_size)
        # slow score
        slow_scorer_size = self.mention_emb_size * 3
        if self.args.use_speaker:
            slow_scorer_size += self.args.feature_dim
        if self.args.use_bucket_offset:
            slow_scorer_size += self.args.feature_dim
        self.slow_pair_scorer = FFNN(self.args.ffnn_depth, slow_scorer_size, self.args.ff_embed_dim, 1,
                                     self.args.ffnn_dropout)
        # speaker, genre (BUT real system doesn't take genre)
        if self.args.use_speaker:
            self.speaker_emb = nn.Embedding(2, self.args.feature_dim)  # 0 not same, 1 same

        if self.args.use_bucket_offset:
            self.bucket_offset_emb = nn.Embedding(10, self.args.feature_dim)
        if self.args.coref_depth > 1:
            self.f_projector = linear(self.mention_emb_size, self.mention_emb_size)


    def forward(self, inputs):
        # get concept reps
        concept_reps = self.embed_scale * self.concept_encoder(inputs['concept'], inputs['concept_char'])
        concept_reps = self.concept_embed_layer_norm(concept_reps)
        if self.args.use_gnn:
            # get graph reps
            mask = torch.ones(1, len(concept_reps)).to(self.args.device)
            graph_data = [concept_reps.transpose(0, 1), mask,
                          inputs['neighbor_index_in'], inputs['edges_index_in'], inputs['mask_in'],
                          inputs['neighbor_index_out'], inputs['edges_index_out'], inputs['mask_out'],
                          inputs['edge_index']]

            concept_graph_reps = self.graph_encoder(graph_data)
        else:
            # remove graph
            concept_graph_reps = concept_reps.transpose(0, 1)

        if self.args.use_token:
            hidden = self.lstm_init_hidden()
            token_reps = self.embed_scale * self.word_encoder(inputs['token'], inputs['token_char'])
            token_reps = self.token_embed_layer_norm(token_reps)
            token_reps, hidden_token = self.lstm(token_reps, hidden)
            token_reps = get_aligment_embed(token_reps.transpose(0, 1), inputs['alignment'], self.args.device)
            concept_graph_reps = torch.cat([concept_graph_reps, token_reps], dim=2)
        # get mention id info
        if self.args.use_gold_cluster:
            mention_ids = inputs['gold_mention_ids']  # [bz = 1, concept]
        elif self.args.use_dict:
            mention_ids = inputs['mention_filter_ids']
        else:
            mention_ids = inputs['mention_ids']

        mention_emb = self.get_mention_embedding(concept_graph_reps, mention_ids)

        mention_emb = self.emb_dropout(mention_emb)

        # use a classifier for implicit role
        if self.args.use_classifier:
            # add ARG information
            arg_classification_logits = self.arg_classification_layer(mention_emb)

            # arg loss
            loss_arg = self.arg_loss(arg_classification_logits.squeeze(dim=0), inputs['concept_class'])

            arg_predicted = torch.argmax(arg_classification_logits, dim=2)
            acc_arg = torch.sum(arg_predicted == inputs['concept_class']).data.tolist() / arg_predicted.size()[1]
            args_embed = self.arg_emb(arg_predicted)
            mention_emb = torch.cat([mention_emb, args_embed], dim=2)
            mention_emb, mention_ids = self.get_arg_classfication_emb(mention_emb, arg_predicted, inputs['concept_class'])




        mention_scores = self.mention_score(mention_emb).squeeze(dim=2)  # [batch = 1, mention]

        # get antecedent info, antecedents: [batch, mention, c]
        # fast_antecedent_scores corresponds to "s_m(i) + s_m(j) + s_pair(i,j)"
        c = min(self.args.antecedent_max_num, mention_ids.shape[1])
        antecedents, antecedent_emb, antecedent_mask, antecedent_offsets, fast_antecedent_scores, antecedents_raw_cpu = \
            self.get_antecedent_info(mention_emb, mention_scores, c)

        # slow_score: s_a(i,j)
        mention_speaker_ids = batch_gather(inputs['speaker'], mention_ids, self.args.device) \
            if self.args.use_speaker else None  # [batch, mention]
        coref_depth = 1 if not self.args.coref_depth else self.args.coref_depth
        assert coref_depth >= 1

        dummy_scores = torch.zeros(self.args.batch_size, mention_ids.shape[1], 1)

        dummy_scores = dummy_scores.to(self.args.device)

        for i in range(coref_depth):
            slow_antecedent_scores = self.get_slow_antecedent_score(mention_emb, mention_speaker_ids,
                                                                    antecedents, antecedent_emb,
                                                                    antecedent_offsets)  # [batch, mention, c]
            antecedent_scores = fast_antecedent_scores + slow_antecedent_scores + \
                                antecedent_mask.float().log()  # [batch, mention, c]

            # merge dummy
            # NaN shouldn't be introduced by F.softmax() because of the ``dummy_scores''
            overall_scores = torch.cat([dummy_scores, antecedent_scores], dim=2)  # [batch, mention, c+1]
            if contain_nan(overall_scores):
                print(overall_scores)
                assert False
            overall_dist = F.softmax(overall_scores, dim=-1)  # [batch, mention, c+1]
            if contain_nan(overall_dist):
                print(overall_dist)
                assert False
            # overall_dist = torch.clamp(F.softmax(overall_scores, dim=-1), 1e-6, 1.0) # [batch, mention, c+1]
            # overall_dist = overall_dist / overall_dist.sum(dim=2, keepdim=True)

            # don't have to calculate the remaining for the last loop
            if i == coref_depth - 1:
                break

            # weighted sum of antecedent embeddings
            overall_emb = torch.cat([mention_emb.unsqueeze(dim=2), antecedent_emb], dim=2)  # [batch, mention, c+1, emb]
            attended_mention_emb = torch.sum(overall_dist.unsqueeze(dim=3) * overall_emb,
                                             dim=2)  # [batch, mention, emb]

            # calculate f
            f = torch.sigmoid(
                self.f_projector(torch.cat([attended_mention_emb, mention_emb], dim=2)))  # [batch, mention, emb]

            # make updates
            mention_emb = f * attended_mention_emb + (1 - f) * mention_emb  # [batch, mention, emb]
            mention_scores = self.mention_score(mention_emb).squeeze(dim=2)  # [batch, mention]
            _, antecedent_emb, _, _, fast_antecedent_scores, _ = \
                self.get_antecedent_info(mention_emb, mention_scores, c)

        overall_dist = clip_and_normalize(overall_dist, 1e-6)
        overall_argmax = torch.argmax(overall_dist, dim=2)  # [batch, mention]
        if self.args.use_gold_cluster:
            mention_cluster_ids = inputs['gold_cluster_ids']  # [batch, mention]
        elif self.args.use_dict:
            mention_cluster_ids = inputs['cluster_filter_ids']
        elif self.args.use_classifier:
            mention_cluster_ids = torch.index_select(inputs['mention_cluster_ids'], 1, mention_ids.squeeze(0))
        else:
            mention_cluster_ids = inputs['mention_cluster_ids']
        antecedent_cluster_ids = batch_gather(mention_cluster_ids, antecedents, self.args.device)
        antecedent_cluster_ids *= antecedent_mask.long()  # [batch, mention, c]

        same_cluster_indicator = antecedent_cluster_ids == mention_cluster_ids.unsqueeze(dim=2)  # [batch, mention, c]
        non_dummy_indicator = (mention_cluster_ids > 0).unsqueeze(dim=2)  # [batch, mention, 1]
        antecedent_labels = same_cluster_indicator & non_dummy_indicator  # [batch, mention, c]
        dummy_labels = ~ (antecedent_labels.any(dim=2, keepdim=True))  # [batch, mention, 1]
        overall_labels = torch.cat([dummy_labels, antecedent_labels], dim=2)  # [batch, mention, c+1]

        loss_coref = -1.0 * torch.sum(overall_dist.log() * overall_labels.float(), dim=2)  # [batch, mention]
        loss_coref = torch.sum(loss_coref, dim=1)  # [batch]

        if self.args.use_classifier:
            loss = loss_coref + loss_arg
            return {'antecedents': antecedents, 'overall_dist': overall_dist,
                    'overall_argmax': overall_argmax,
                    'loss_coref': torch.mean(loss_coref),
                    'loss_arg': torch.mean(loss_arg),
                    'acc_arg': acc_arg,
                    'loss': torch.mean(loss),
                    'mention_ids': mention_ids,
                    'mention_cluster_ids': mention_cluster_ids,
                    'antecedents_raw_cpu': antecedents_raw_cpu}
        else:
            loss = loss_coref
            return {'antecedents': antecedents, 'overall_dist': overall_dist,
                    'overall_argmax': overall_argmax,
                    'loss': torch.mean(loss),
                    'antecedents_raw_cpu': antecedents_raw_cpu}

    # mention_emb: [batch, mention, emb]
    # mention_scores: [batch, mention]
    # mention_mask: [batch, mention]
    # c: scalor
    def get_antecedent_info(self, mention_emb, mention_scores, c):
        batch_size, mention_num, emb_size = list(mention_emb.size())

        antecedent_offsets = torch.arange(1, c + 1).view(1, 1, c).expand(batch_size, mention_num, -1)
        antecedents_raw_cpu = torch.arange(mention_num).view(1, mention_num, 1).expand(batch_size, -1, c) - \
                              antecedent_offsets  # [batch=1, mention, c]
        antecedents = torch.clamp(antecedents_raw_cpu, 0, mention_num - 1)
        antecedent_mask = antecedents_raw_cpu >= 0

        antecedent_mask = antecedent_mask.to(self.args.device)
        antecedent_offsets = antecedent_offsets.to(self.args.device)
        antecedents = antecedents.to(self.args.device)

        # Part 1: s_m(i) + s_m(j)
        fast_antecedent_scores_1 = batch_gather(mention_scores, antecedents, self.args.device) + \
                                   mention_scores.unsqueeze(dim=2)  # [batch, mention, c]

        antecedent_emb = batch_gather(mention_emb, antecedents, self.args.device)  # [batch, mention, c, emb]

        ## Part 2:
        # source_emb = self.dropout(self.fast_src_projector(antecedent_emb).view(batch_size * mention_num,
        #    c, emb_size)) # [batch * mention, c, emb]
        # target_emb = self.dropout(mention_emb.view(batch_size * mention_num, emb_size, 1)) # [batch * mention, emb, 1]
        # assert utils.shape(source_emb, 0) == utils.shape(target_emb, 0)
        # fast_antecedent_scores_2 = torch.matmul(source_emb, target_emb).view(batch_size, mention_num, c) # [batch * mention, c]

        fast_antecedent_scores = fast_antecedent_scores_1  # + fast_antecedent_scores_2

        return antecedents, antecedent_emb, antecedent_mask, antecedent_offsets, fast_antecedent_scores, antecedents_raw_cpu

    # s_a(i,j) = FFNN([g_i,g_j,g_i*g_j,\phi(i,j)])
    def get_slow_antecedent_score(self, mention_emb, mention_speaker_ids,
                                  antecedents, antecedent_emb, antecedent_offsets):
        batch_size, mention_num, c = list(antecedents.size())
        feature_emb_list = []

        if self.args.use_speaker:
            antecedent_speaker_ids = batch_gather(mention_speaker_ids, antecedents, self.args.device)
            same_speaker = (
                    antecedent_speaker_ids == mention_speaker_ids.unsqueeze(dim=2)).long()  # [batch, mention, c]
            same_speaker_emb = self.speaker_emb(same_speaker)  # [batch, mention, c, emb]
            feature_emb_list.append(same_speaker_emb)

        if self.args.use_bucket_offset:
            antecedent_offset_buckets = self.bucket_distance(antecedent_offsets)
            antecedent_offset_emb = self.bucket_offset_emb(antecedent_offset_buckets)  # [batch, mention, c, emb]
            feature_emb_list.append(antecedent_offset_emb)

        feature_emb = self.emb_dropout(torch.cat(feature_emb_list, dim=3))  # [batch, mention, c, embemb]
        target_emb = mention_emb.unsqueeze(dim=2).expand(-1, -1, c, -1)  # [batch, mention, 1, emb]
        similarity_emb = antecedent_emb * target_emb
        pair_emb = torch.cat([target_emb, antecedent_emb, similarity_emb, feature_emb],
                             dim=3)  # [batch, mention, c, emb]

        slow_antecedent_scores = self.slow_pair_scorer(pair_emb).squeeze(dim=-1)
        return slow_antecedent_scores

    # embeddings: [bz=1, seq_len, emb]
    # mention_starts, mention_ends and mention_mask: [batch, mentions]
    # s_m(i) = FFNN(g_i)
    # g_i = [x_i^start, x_i^end, x_i^head, \phi(i)]
    def get_mention_embedding(self, embeddings, mention_ids):
        mention_emb_list = []
        mention_start_emb = batch_gather(embeddings, mention_ids, self.args.device)  # [batch, mentions, emb]
        mention_emb_list.append(mention_start_emb)
        return torch.cat(mention_emb_list, dim=2)


    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = torch.floor(distances.float().log() / math.log(2)).long() + 3
        use_identity = (distances <= 4).long()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9).long()

    def get_arg_classfication_emb(self, mention_emb, arg_predicted, gold_label):
        # a=1
        label = arg_predicted.tolist()[0]
        gold_label = gold_label.tolist()
        index, gold_index = [], []
        for i, l in enumerate(label):
            if l > 0:
                index.append(i)
        for i, l in enumerate(gold_label):
            if l > 0:
                gold_index.append(i)
        if self.training:
            index = torch.tensor(gold_index).to(self.args.device)
        else:
            if len(index) == 0:
                # index = torch.tensor(gold_index).to(self.args.device)
                # print('xxxxxxxxxxx')
                index = torch.tensor(label).to(self.args.device)
            else:
                index = torch.tensor(index).to(self.args.device)
        emb = torch.index_select(mention_emb, 1, index)
        return emb, index.unsqueeze(0)


    def lstm_init_hidden(self):

        result = (torch.zeros(2*self.args.bilstm_layer_num, 1, self.args.bilstm_hidden_dim, requires_grad=True).to(self.args.device),
                  torch.zeros(2*self.args.bilstm_layer_num, 1, self.args.bilstm_hidden_dim, requires_grad=True).to(self.args.device))
        return result
