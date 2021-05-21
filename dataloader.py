from operator import itemgetter
from config import parse_config
import argparse
from vocab import Vocab, STR, END, SEP, C2T, PAD
import json
import torch
import random
from collections import Counter


def get_align_mapping(alignments, token_lens):
    align_mapping = []

    for i, align in enumerate(alignments):
        a = []
        if i == 0:
            for k in align:
                if type(k) == type(["sean"]):
                    temp = []
                    for kk in k:
                        temp.append(int(kk))
                    a.append(temp)
                else:

                    a.append(k)
        else:
            for k in align:
                if type(k) == type(["sean"]):
                    add_index = sum(token_lens[:i])
                    temp = []
                    for kk in k:
                        temp.append(int(kk) + add_index)
                    a.append(temp)
                else:
                    if k != -1:
                        add_index = sum(token_lens[:i])
                        a.append(k + add_index)
                    else:
                        a.append(k)
        # align_mapping.append(a)
        align_mapping.extend(a)

    return align_mapping


def get_edge_mapping(edges, concept_lens):
    edges_mapping = []

    root_index = [sum(concept_lens[:i]) for i in range(len(concept_lens))]
    for i, es in enumerate(edges):
        for j, e in enumerate(es):
            edges_mapping.append([e[0], e[1]+root_index[i], e[2]+root_index[i]])
    # add full connect root_node
    root_edge_type = 'AMR_ROOT'
    for i in root_index:
        for j in root_index:
            edges_mapping.append([root_edge_type, i, j])

    return edges_mapping


def get_cluster_mapping(clusters, concept_lens):
    cluster_mapping = []
    cluster_mapping_labels = []
    for i, cluster in enumerate(clusters):
        cluster_mapping.append([])
        cluster_mapping_labels.append([])
        for j, c in enumerate(cluster):
            cluster_mapping[-1].append((sum(concept_lens[:c[0]]) + c[1]))
            cluster_mapping_labels[-1].append(c[2])
    return cluster_mapping, cluster_mapping_labels


def get_concept_labels(cluster, cluster_labels, concepts):
    concept_labels = []
    cluster = [item for sublist in cluster for item in sublist]
    cluster_labels = [item for sublist in cluster_labels for item in sublist]
    for i in range(len(concepts)):
        if i in cluster:
            concept_labels.append(cluster_labels[cluster.index(i)])
        else:
            concept_labels.append(-2)
    a = [i + 2 for i in concept_labels]
    return a


def get_bert_ids(tokens, args, tokenizer):

    sentence_ids, sentence_toks, sentence_lens = [], [], []
    for si, sentence in enumerate(tokens):
        sent_len = 0
        for word in sentence:
            for char in tokenizer.tokenize(word):
                sentence_ids.append(si)
                sentence_toks.append(char)
                break
        sentence_lens.append(sent_len)
    sentence_toks = [x if x in tokenizer.vocab else '[UNK]' for x in sentence_toks]
    input_ids = tokenizer.convert_tokens_to_ids(sentence_toks)

    return input_ids


def get_speaker(id_info):
    speaker = None
    doc_type = id_info.split("::doc_type")[1].strip()
    if doc_type == "dfa":
        p = id_info.split("::post")[1]
        if p[:2] == "  ":
            temp = 'unk'
        elif p[:1] == " ":
            temp = p.split()[0]
        else:
            assert False
        speaker = temp
    elif doc_type == "dfb":
        p = id_info.split("::speaker")[1]
        if p[:2] == "  ":
            temp = 'unk'
        elif p[:1] == " ":
            temp = p.split()[0]
        else:
            assert False
        speaker = temp
    else:
        speaker = 'unk'
    return speaker

def load_json(file_name, args, tokenizer):
    with open(file_name, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    doc_data = []
    for i , doc in enumerate(json_dict):

        # 在这个doc里， 先把句子合起来
        toks, concepts, alignments, edges, clusters = [], [], [], [], []
        tokens = []
        speakers = []
        concept_lens, token_lens, token_seps_index = [], [], []
        concepts_for_align = []
        for j, inst in enumerate(doc['data']):
            # get snts, tokens, concepts

            toks.extend(inst['token'].split())
            tokens.append(inst['token'].split())
            concept_lens.append(inst['concept_len'])
            speaker = get_speaker(inst['id_info'])

            speakers.extend([speaker] * inst['concept_len'])
            concepts_for_align.append(inst['concept'])
            token_lens.append(inst['token_len'])
            alignments.append(inst['alignment'])
            edges.append(inst['edge'])
        concepts = [y for x in concepts_for_align for y in x]
        a = sum(concept_lens)
        token_bert_ids = get_bert_ids(tokens, args, tokenizer)
        # get alignments
        align_mapping = get_align_mapping(alignments, token_lens)
        # assert len(align_mapping) == len(concepts)
        # get edge mapping
        edge_mapping = get_edge_mapping(edges, concept_lens)
        # get cluster mapping
        cluster_mapping, cluster_mapping_labels = get_cluster_mapping(doc['cluster'], concept_lens)
        # -2, -1, 0, 1, 2
        concept_labels = get_concept_labels(cluster_mapping, cluster_mapping_labels, concepts)
        doc_data.append([speakers, toks, token_bert_ids, concepts, align_mapping,
                         edge_mapping, cluster_mapping, concept_labels, token_lens])

    return doc_data


def make_vocab(batch_data, char_level=False):
    count = Counter()
    for seq in batch_data:
        count.update(seq)
    if not char_level:
        return count
    char_cnt = Counter()
    for x, y in count.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return count, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n' % (x, y))


def preprocess_vocab(train_data, args):

    # batch data not make batch
    tokens, concepts, relations = [], [], []
    for i, doc in enumerate(train_data):
        tokens.append(doc[1])
        concepts.append(doc[3])
        temp = []
        for j, rel in enumerate(doc[5]):
            temp.append(rel[0])
        relations.append(temp)
    a = 1

    # make vocab
    token_vocab, token_char_vocab = make_vocab(tokens, char_level=True)
    concept_vocab, concept_char_vocab = make_vocab(concepts, char_level=True)
    relation_vocab = make_vocab(relations, char_level=False)
    write_vocab(token_vocab, args.token_vocab)
    write_vocab(token_char_vocab, args.token_char_vocab)
    write_vocab(concept_vocab, args.concept_vocab)
    write_vocab(concept_char_vocab, args.concept_char_vocab)
    write_vocab(relation_vocab, args.relation_vocab)


def list_to_tensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data


def list_string_to_tensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data


def get_graph(nodes, edges):
    neighbor_num_in = []
    edges_in = []
    edges_out = []
    neighbor_num_out = []
    neighbors_in = []
    neighbors_out = []
    edge_index = []
    for i, e in enumerate(edges):
        edge_index.append(e[1:])
    for n in range(len(nodes)):
        count, count_in, count_out = 0, 0, 0
        neighbors_per_node_in = []
        neighbors_per_node_out = []
        edges_per_node_in = []
        edges_per_node_out = []
        for i, e in enumerate(edges):
            if n in e:
                count = count + 1
                if e[1] == n:
                    count_out = count_out + 1
                    neighbors_per_node_out.append(e[2])
                    edges_per_node_out.append(e[0])
                else:
                    count_in = count_in + 1
                    neighbors_per_node_in.append(e[1])
                    edges_per_node_in.append(e[0])
        neighbor_num_in.append(count_in)
        neighbor_num_out.append(count_out)
        neighbors_in.append(neighbors_per_node_in)
        neighbors_out.append(neighbors_per_node_out)
        edges_in.append(edges_per_node_in)
        edges_out.append(edges_per_node_out)
    max_neighbor_num_in = max(neighbor_num_in)
    max_neighbor_num_out = max(neighbor_num_out)
    mask_in = [[1] * max_neighbor_num_in for x in range(len(edges_in))]
    mask_out = [[1] * max_neighbor_num_out for x in range(len(edges_out))]
    for i, e in enumerate(edges_in):
        mask_in[i][len(e):max_neighbor_num_in] = [0] * (max_neighbor_num_in - len(e))
        neighbors_in[i].extend([-1] * (max_neighbor_num_in - len(e)))
        edges_in[i].extend([PAD] * (max_neighbor_num_in - len(e)))
    for i, e in enumerate(edges_out):
        mask_out[i][len(e):max_neighbor_num_out] = [0] * (max_neighbor_num_out - len(e))
        neighbors_out[i].extend([-1] * (max_neighbor_num_out - len(e)))
        edges_out[i].extend([PAD] * (max_neighbor_num_out - len(e)))
    graph = {
        "edge_index": edge_index,
        "neighbor_index_in": neighbors_in,
        "neighbor_index_out": neighbors_out,
        "edges_in": edges_in,
        "edges_out": edges_out,
        "mask_in": mask_in,
        "mask_out": mask_out
    }
    return graph


def build_graph(data, vocabs, token2concept=False):
    if token2concept:
        new_nodes = data[3] + data[2]
        new_edges = data[5]
        for i, j in enumerate(data[3]):
            if isinstance(j, int):
                new_edges.append([C2T, i, j + len(data[2])])
            else:
                for k in j:
                    new_edges.append([C2T, i, k + len(data[2])])
        graph_data = get_graph(new_nodes, new_edges)
        return graph_data
    else:
        nodes = data[3]
        edges = data[5]
        graph_data = get_graph(nodes, edges)
        return graph_data


def get_cluster(clusters):
    # remove same concept in one cluster and remove same concept in different clusters
    clusters_filter1 = []
    for i, c in enumerate(clusters):
        if len(c) == len(set(c)):
            clusters_filter1.append(c)
        else:
            if len(set(c)) > 1:
                clusters_filter1.append(list(set(c)))
            else:
                continue

    clusters_filter2, cs = [], []
    for i, c in enumerate(clusters_filter1):
        if i == 0:
            clusters_filter2.append(c)
            cs.extend(c)
        else:
            t = []
            for cc in c:
                if cc in cs:
                    continue
                else:
                    t.append(cc)
            if len(t) > 1:
                cs.extend(t)
                clusters_filter2.append(t)

    cluster = []
    for i, c in enumerate(clusters_filter2):
        # for j in set(c):
        # assert len(c) == len(set(c))
        # if len(c) != len(set(c)):
        #     print('xx')
        assert len(c) == len(set(c))
        for j in c:
            cluster.append([j, i + 1])
    temp = sorted(cluster, key=itemgetter(0))
    c = [i[0] for i in temp]
    c_ids = [i[1] for i in temp]
    return c, c_ids


def data_to_device_evl(args, train_data):
    for j, data in enumerate(train_data):
        features = []
        for i, d in enumerate(data):
            if d == 'concept_len' or d == 'token_segments' \
                    or d == 'alignment' or d == 'concept4filter':
                continue
            else:
                train_data[j][d] = train_data[j][d].to(args.device)
    return train_data


def data_to_device(args, evl_data):

    features = []
    for i, data in enumerate(evl_data):
        if data == 'concept_len' or data == 'token_segments' \
                or data == 'alignment' or data == 'concept4filter':
            continue
        else:
            evl_data[data] = evl_data[data].to(args.device)
    return evl_data


def pre_speaker(speakers):
    speaker_ids = []
    speaker_dict = {'unk': 0, '[SPL]': 1}
    for s in speakers:
        speaker_dict[s] = len(speaker_dict)
    for s in speakers:
        speaker_ids.append(speaker_dict[s])

    return speaker_ids


def get_filter_ids(args, concept, concept_class, mention_ids, mention_cluster_ids):
    with open(args.dict_file, 'r', encoding='utf') as f:
        dict_file = [line.strip('\n') for line in f]
    dict_file = dict_file[:args.dict_size]
    mention_filter_ids, cluster_filter_ids, concept_labels = [], [], []
    for i, c in enumerate(concept):
        if c not in dict_file:
            mention_filter_ids.append(mention_ids[i])
            cluster_filter_ids.append(mention_cluster_ids[i])
            concept_labels.append(concept_class[i])
        else:
            continue
    return mention_filter_ids, cluster_filter_ids, concept_labels


def data_to_feature(args, train_data, vocabs):

    features = []
    for i, data in enumerate(train_data):
        #print(i)
        if i == 130:
            a = 1
        item = dict()
        # concept
        item['concept_len'] = len(data[3])
        item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
        item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])

        # speaker
        item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)
        # graph
        graph = build_graph(data, vocabs, False)

        item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
        item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
        item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
        item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
        item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
        item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
        item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
        # token
        token_len = len(data[1])
        item['token_len'] = torch.LongTensor([token_len])
        item['token'] = list_to_tensor([data[1]], vocabs['token'])
        item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
        item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
        item['token_segments'] = data[-1]
        # cluster
        cluster, cluster_ids = get_cluster(data[6])
        mention_cluster_ids = [0] * item['concept_len']
        mention_ids = list(range(item['concept_len']))
        for idx, (mention_id, cluster_id) in enumerate(zip(cluster, cluster_ids)):
            mention_cluster_ids[mention_id] = cluster_id
        item['gold_mention_ids'] = torch.LongTensor(cluster).unsqueeze(0)
        item['gold_cluster_ids'] = torch.LongTensor(cluster_ids).unsqueeze(0)
        item['mention_ids'] = torch.LongTensor(mention_ids).unsqueeze(0)
        item['mention_cluster_ids'] = torch.LongTensor(mention_cluster_ids).unsqueeze(0)
        # alignment
        item['alignment'] = data[4]
        # dict to filter
        # item['concept4filter'] = data[3]

        mention_filter_ids, cluster_filter_ids, concept_labels = get_filter_ids(args, data[3], data[7], mention_ids, mention_cluster_ids)
        item['mention_filter_ids'] = torch.LongTensor(mention_filter_ids).unsqueeze(0)
        item['cluster_filter_ids'] = torch.LongTensor(cluster_filter_ids).unsqueeze(0)

        if args.use_dict:
            item['concept_class'] = torch.LongTensor(concept_labels)
        else:
            item['concept_class'] = torch.LongTensor(data[7])

        features.append(item)

    return features

def make_data_evl(args, tokenizer):

    # load vocab
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 0, None)
    vocabs['token'] = Vocab(args.token_vocab, 0, [STR, END, SEP])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 0, None)
    vocabs['token_char'] = Vocab(args.token_char_vocab, 0, None)
    vocabs['relation'] = Vocab(args.relation_vocab, 1, [C2T])

    # make batch, batch_size = 1
    test_data = load_json(args.test_data, args, tokenizer)
    test_features = data_to_feature(args, test_data, vocabs)
    return test_features, vocabs



def make_data(args, tokenizer):
    # make vocab,
    print("load train data")
    train_data = load_json(args.train_data, args, tokenizer)
    preprocess_vocab(train_data, args)
    # load vocab
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 0, None)
    vocabs['token'] = Vocab(args.token_vocab, 0, [STR, END, SEP])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 0, None)
    vocabs['token_char'] = Vocab(args.token_char_vocab, 0, None)
    vocabs['relation'] = Vocab(args.relation_vocab, 1, [C2T])
    for name in vocabs:
        print((name, vocabs[name].size, vocabs[name].coverage))
    # make batch, batch_size = 1
    dev_data = load_json(args.dev_data, args, tokenizer)
    test_data = load_json(args.test_data, args, tokenizer)
    train_features = data_to_feature(args, train_data, vocabs)
    dev_features = data_to_feature(args, dev_data, vocabs)
    test_features = data_to_feature(args, test_data, vocabs)
    return train_features, dev_features, test_features, vocabs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_config(parser)
    # add
    parser.add_argument("--model_path", default='ckpt/models')
    args = parser.parse_args()

    pre_data = make_data(args)
    print('Done!')
