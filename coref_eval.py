
import random
import math
import numpy as np
import argparse
import os, sys, json, codecs
import collections
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from coref_model import *
from metrics import *
from config import *
from dataloader import *

def get_predicted_antecedents(antecedents, overall_argmax, mention_num):
    predicted_antecedents = []
    for i, index in enumerate(overall_argmax):
        if i >= mention_num:
            continue
        predicted_antecedents.append(-1 if index < 0 else antecedents[i, index])
        if index >= 0:
            assert antecedents[i, index] >= 0
    return predicted_antecedents


# mention_starts, mention_ends: ndarray of [mention], token position
# predicted_antecedents: list of [mention], mention id
def get_predicted_clusters(mention_ids, mention_num, predicted_antecedents):
    mention_to_predicted_cluster = {} # dict[mention span] --> cluster id
    predicted_clusters = [] # list[cluster], cluster: list of mention
    for cur_index in range(mention_num):
        predicted_index = predicted_antecedents[cur_index]
        if predicted_index < 0:
            continue
        assert cur_index > predicted_index, (cur_index, predicted_index)
        #
        predicted_antecedent = (mention_ids[predicted_index], mention_ids[predicted_index])
        if predicted_antecedent in mention_to_predicted_cluster:
            predicted_cluster_index = mention_to_predicted_cluster[predicted_antecedent]
        else:
            predicted_cluster_index = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted_cluster[predicted_antecedent] = predicted_cluster_index
        # predited --> cur ???
        mention = (mention_ids[cur_index], mention_ids[cur_index])
        predicted_clusters[predicted_cluster_index].append(mention)
        mention_to_predicted_cluster[mention] = predicted_cluster_index

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m:predicted_clusters[i] for m,i in mention_to_predicted_cluster.items()}
    return predicted_clusters, mention_to_predicted


def evaluate_coref(mention_ids, mention_num, mention_clusters, predicted_antecedents, evaluator):
    gold_clusters = collections.defaultdict(list)
    for i in range(mention_num):
        ci = mention_clusters[i]
        if ci > 0:
            mention = (mention_ids[i], mention_ids[i])
            gold_clusters[ci].append(mention)

    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters.values()]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            #assert mention not in mention_to_gold
            mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = get_predicted_clusters(mention_ids, mention_num, predicted_antecedents)

    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters


def evaluate(model, dataset, log_file, args, output_path=None, reference_path=None):
    model.eval()

    random.shuffle(dataset)
    losses, losses_coref, losses_arg, accs = [], [], [], []
    coref_evaluator = CorefEvaluator()
    for step, ori_batch in enumerate(dataset):
        batch = {k: v.to(args.device) if type(v) == torch.Tensor else v \
                for k, v in ori_batch.items()}
        if len(batch['concept']) > 500:
            continue
        outputs = model(batch)
        loss = outputs['loss']
        if args.use_classifier:
            losses_coref.append(outputs['loss_coref'])
            losses_arg.append(outputs['loss_arg'])
            accs.append(outputs['acc_arg'])
        losses.append(loss.item() if type(loss) == torch.Tensor else loss)

        # doc_keys.extend(ori_batch['doc_keys'])
        if args.use_gold_cluster:
            mention_ids = ori_batch['gold_mention_ids'].cpu().numpy()  # [batch, mention]
            mention_nums = mention_ids.size  # [batch]
            mention_clusters = ori_batch['gold_cluster_ids'].cpu().numpy() # [batch, mention]
        elif args.use_dict:
            mention_ids = ori_batch['mention_filter_ids'].cpu().numpy()  # [batch, mention]
            mention_nums = mention_ids.size  # [batch]
            mention_clusters = ori_batch['cluster_filter_ids'].cpu().numpy() # [batch, mention]
        elif args.use_classifier:
            mention_ids = outputs['mention_ids'].cpu().numpy()
            mention_nums = mention_ids.size
            mention_clusters = outputs['mention_cluster_ids'].cpu().numpy()
        else:
            mention_ids = ori_batch['mention_ids'].cpu().numpy()  # [batch, mention]
            mention_nums = mention_ids.size  # [batch]
            mention_clusters = ori_batch['mention_cluster_ids'].cpu().numpy()  # [batch, mention]
        antecedents = outputs['antecedents_raw_cpu'].numpy() # [batch, mention, c]
        overall_argmax = outputs['overall_argmax'].detach().cpu().numpy() - 1 # [batch, mention]
        assert mention_ids.shape == overall_argmax.shape
        batch_size = 1
        for i in range(batch_size):
            predicted_antecedents = get_predicted_antecedents(antecedents[i],
                    overall_argmax[i], mention_nums)
            predicted_clusters = evaluate_coref(mention_ids[i], mention_nums, mention_clusters[i], predicted_antecedents, coref_evaluator)

    p, r, f = coref_evaluator.get_prf()
    print('----Evaluate loss: {:.3f}  Coref loss: {:.3f}  ARG loss: {:.3f}  ARG acc: {:.3f}'
          .format(mean(losses), mean(losses_coref), mean(losses_arg), mean(accs)))
    print("----Average F1 (py): {:.3f}% precision (py): {:.3f}% recall (py): {:.3f}% on {} docs"
          .format(f * 100, p * 100, r * 100, len(dataset)))

    log_file.write("Average F1 (py): {:.2f}% on {} docs\n".format(f * 100, len(dataset)))
    log_file.write("Average precision (py): {:.2f}%\n".format(p * 100))
    log_file.write("Average recall (py): {:.2f}%\n".format(r * 100))

    return f

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_config(parser)
    parser.add_argument('--prefix_path', type=str, default='./ckpt/coref.amr')

    args = parser.parse_args()
    log_file = sys.stdout
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    test_data, vocabs = make_data_evl(args, tokenizer)
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))
    print("Num test examples = {}".format(len(test_data)))
    print("Num test batches = {}".format(len(test_data)))
    print('Compiling model...')
    model = AMRCorefModel.from_pretrained(args.bert_tokenizer_path, args, vocabs)

    print('Loading the model...')
    model.load_state_dict(torch.load(args.prefix_path + ".model"))
    model.to(args.device)
    test_data = data_to_device(args, test_data)
    print('Decoding...')
    evaluate(model, test_data, log_file, args)

