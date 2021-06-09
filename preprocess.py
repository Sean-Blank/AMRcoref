import re
import json
import numpy as np
import string
from xml.etree import ElementTree as ET
import os
from AMRGraph import _is_abs_form
from AMRGraph import AMRGraph
from smatch.amr import AMR
#
# test: dfa 007, 027, 041, 063, 077, 081, 093, 095, 134
# dev:  dfa 006, 028, 042, 064, 077, 082, 094, 096, 135
amr_path = 'data/amr-unsplit/'
xml_path = 'data/xml-unsplit/'
align_path = 'data/align_unsplit/'


class AMRIO_align:
    def __init__(self):
        pass
    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::tok '):
                    tokens = line[len('# ::tok '):]
                elif line.startswith('# ::alignments'):
                    if len(line) == len('# ::alignments'):
                        alignments = ["None"]
                    else:
                        alignments = line[len('# ::alignments '):]
                    graph_line = AMR.get_amr_line(f)
                    amr = AMR.parse_AMR_line(graph_line)
                    myamr = AMRGraph(amr)
                    yield tokens, alignments, amr, myamr


class AMRIO:
    def __init__(self):
        pass
    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                elif line.startswith('# ::save-date '):
                    other = line[len('# ::save-date '):]
                    graph_line = AMR.get_amr_line(f)
                    amr = AMR.parse_AMR_line(graph_line)
                    myamr = AMRGraph(amr)
                    yield amr_id, sentence, graph_line, amr, myamr


def get_xml_data(parser):
    rels=[]
    for ident in parser.getroot()[1][0]:
        rels.append([])
        rels[-1].append(ident.attrib['relationid'])
        for ment in ident:
            if ment.tag == 'mention':
                rels[-1].append(ment.attrib)
                rels[-1][-1]['tag'] = 'mention'
                # rels[-1][-1]['id'] = re.findall("\d+", rels[-1][-1]['id'])[-1]
                rels[-1][-1]['id'] = rels[-1][-1]['id']
            elif ment.tag == 'implicitrole':
                rels[-1].append(ment.attrib)
                rels[-1][-1]['tag'] = 'implicit_' + ment.attrib['argument'][-1]
                # rels[-1][-1]['id'] = re.findall("\d+", rels[-1][-1]['id'])[-1]
                rels[-1][-1]['id'] = rels[-1][-1]['id']

    for ident in parser.getroot()[1][1]:
        rels.append([])
        rels[-1].append(ident.attrib['relationid'])
        for ment in ident:
            if ment.tag == 'mention':
                rels[-1].append(ment.attrib)
                rels[-1][-1]['tag'] = 'mention'
                rels[-1][-1]['id'] = re.findall("\d+", rels[-1][-1]['id'])[-1]
            elif ment.tag == 'implicitrole':
                rels[-1].append(ment.attrib)
                rels[-1][-1]['tag'] = 'implicit'
                rels[-1][-1]['id'] = re.findall("\d+", rels[-1][-1]['id'])[-1]

    for ident in parser.getroot()[1][2]:
        rels.append([])

        rels[-1].append('set' + ident.attrib['relationid'][3:])
        for i, ment in enumerate(ident):
            if i ==0:
                rels[-1].append(ment.attrib)
                rels[-1][-1]['tag'] = 'super'
            else:
                rels[-1].append(ment.attrib)
                rels[-1][-1]['tag'] = 'member'
    # 这次只保留rel
    rels_rel = []
    for i, rel in enumerate(rels):
        if rel[0][0:3] == "rel":
            rels_rel.append(rel)

    # # 补全，有一些数据有缺失rel,重排序
    # rel_num = len(rels_rel)
    # for i in range(rel_num):
    #     rels_rel[i][0] = "rel-" + str(i)

    # sort
    # for i, ele in enumerate(rels_rel):
    #     rels_rel[i].append(int(ele[0].split('-')[1]))
    #
    # def take_last(rels_rel):
    #     return ele[-1]
    # rels_rel.sort(key=take_last)
    # for i in range(len(rels_rel)):
    #     rels_rel[i].pop(-1)
    return rels_rel


def read_file_align(filename):
    # read preprocessed amr file
    token, lemma, abstract, amrs, myamrs = [], [], [], [], []
    for tok, align, amr, myamr in AMRIO_align.read(filename):
        token.append(tok)
        amrs.append(amr)
        myamrs.append(myamr)
    # print ('read from %s, %d amrs'%(filename, len(token)))
    return token, amrs, myamrs


def read_file_raw(filename):
    # read preprocessed amr file
    snts, ids, amr_lines, amrs, myamrs = [], [], [], [], []
    for amr_id, sentence, amr_line, amr, myamr in AMRIO.read(filename):
        ids.append(amr_id.split('::date')[0])
        snts.append(sentence)
        amrs.append(amr)
        myamrs.append(myamr)
        amr_lines.append(amr_line)
    # print ('read from %s, %d amrs'%(filename, len(token)))
    return ids, snts, amr_lines, amrs, myamrs


def get_align_index(align_file):
    align_index = []
    concept_raw = []
    v2c = []
    tokens, amrs, myamrs = read_file_align(align_file)
    for i, (token, amr, myamr) in enumerate(zip(tokens, amrs, myamrs)):
        # deep first
        #
        concept_ordered, _, _, _, nodes = myamr.collect_concepts_and_relations()
        v2c.append(nodes)
        # print(i)
        # concept = amr.node_values
        # assert concept_ordered == concept
        concept_raw.append([])
        align_index.append([])

        for c in concept_ordered:

            if "~" in c:
                concept_raw[-1].append(c.split('~')[0])
                digits = c.split('~')[1].split('.')[1]
                if "," in digits:
                    align_index[-1].append(digits.split(','))
                else:
                    align_index[-1].append(int(digits))
            else:
                concept_raw[-1].append(c)
                align_index[-1].append(-1)
    return align_index, tokens, concept_raw, v2c


def get_amrs_raw_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        amrs = []
        for line in f:
            line = line.rstrip()
            if line == '':
                amrs.append([])
            elif line.startswith('# ::'):
                amrs[-1].append(line)
            else:
                if line.startswith('('):
                    amrs[-1].append([])
                    amrs[-1][-1].append(line)
                else:
                    amrs[-1][-1].append(line)
    while len(amrs[-1]) == 0:
        amrs.pop()
    return amrs


def get_data(align_index, concept_raw, tokens, raw_file, v2c):
    amrs_raw = get_amrs_raw_data(raw_file)
    data = []
    ids, snts, amr_lines, amrs, myamrs = read_file_raw(raw_file)
    for i, (id, snt, amr_line, amr, align, token, amr_raw, con_raw, vc) in enumerate(zip(
            ids, snts, amr_lines, amrs, align_index, tokens, amrs_raw, concept_raw, v2c)):
        data.append([])
        # concept, depth, relation, connected = amr.collect_concepts_and_relations()
        # assert connected, "not connected"
        data[-1].append(id)
        data[-1].append(snt)
        data[-1].append(token)
        data[-1].append(align)
        data[-1].append(amr_line)
        data[-1].append(amr_raw[3])
        data[-1].append(con_raw)
        data[-1].append(vc)
    return data



def preprocess(parser, file_name):

    # step 1: read xml data, get doc ids
    relations = get_xml_data(parser)
    # step 2: read raw & align_unsplit amr text
    align_file = align_path + file_name + '.align'
    align_index, tokens, concept_raw, v2c = get_align_index(align_file)
    raw_file = amr_path + file_name + '.txt'
    # step 3: get the final data
    data_per_doc = get_data(align_index, concept_raw, tokens, raw_file, v2c)

    return data_per_doc, relations
    # print()


def mapping_edges(concept, amr, v2c):
    edges = []
    instance_triple, attribute_triple, relation_triple = amr.get_triples()
    assert len(amr.nodes) == len(concept) == len(amr.node_values)
    for i, triple in enumerate(relation_triple):
        source_node = v2c.index(triple[1])
        target_node = v2c.index(triple[2])
        edges.append([triple[0], source_node, target_node])
    return edges


def get_clusters_info(links, data):
    cluster = []
    amr_ids = []
    # step1: 对于每个句子，统计一下token长度，amr节点数量
    for i in range(len(data)):
        # 节点数量 speaker
        amr_ids.append(data[i][0].split('::speaker')[0].strip())
        data[i].append(len(data[i][3]))
        # 句子token数
        data[i].append(len(data[i][2].split(' ')))
        amr = AMR.parse_AMR_line(data[i][4])
        # myamr = AMRGraph(amr)
        # concept = [x.split('~')[0] for x in amr.node_values]
        concept = data[i][6]
        # data[i].append(concept)
        # 根据生成的concept序列（深度或广度）映射边的关系
        edge_index = mapping_edges(concept, amr, data[i][7])
        data[i].append(edge_index)
        a = 1
    # step2: 把所有节点排列起来，
    # step2: 对于每条链，统计出它在当前amr图中的信息：第几句中的第几个节点，标签是什么。【】
    for i, link in enumerate(links):
        cluster.append([])
        for j, mention in enumerate(link[1:]):
            snt_id = amr_ids.index(mention['id'])
            if mention['tag'] == 'mention':
                tag = -1
                #node_id = AMR.parse_AMR_line(data[snt_id][4]).nodes.index(mention['variable'])
                node_id = data[snt_id][7].index(mention['variable'])
                cluster[-1].append([snt_id, node_id, tag])
                # node_id = data[snt_id][9].index(mention['concept'])
            elif mention['tag'] == 'implicit_0':
                tag = 0
                node_id = data[snt_id][7].index(mention['parentvariable'])
                cluster[-1].append([snt_id, node_id, tag])
            elif mention['tag'] == 'implicit_1':
                tag = 1
                node_id = data[snt_id][7].index(mention['parentvariable'])
                cluster[-1].append([snt_id, node_id, tag])
            elif mention['tag'] == 'implicit_2':
                tag = 2
                node_id = data[snt_id][7].index(mention['parentvariable'])
                cluster[-1].append([snt_id, node_id, tag])
            else:
                continue
            # cluster[-1].append([snt_id, node_id, tag])
    a= 1
    # remove some singleton concept caused by ARG3 and ARG4
    c_remove_args = []
    for i, c in enumerate(cluster):
        if len(c) > 1:
            c_remove_args.append(c)
    return data, c_remove_args


def pre_to_json(data, links, file_name):
    doc_data = []

    data, cluster = get_clusters_info(links, data)
    # remove some singleton concept caused by ARG3 and ARG4

    # 预处理完成，合并操作，这里amr图如何合成大图是个问题：
    # 这里直接并起来，暂时不考虑加全局节点
    a = 1

    data_pre = []
    for i, line in enumerate(data):
        temp = {
            'id_info': line[0],
            'token': line[2],
            'token_len': line[9],
            'amr': line[4],
            'concept': line[6],
            'concept_len': line[8],
            'alignment': line[3],
            'edge': line[10]
        }
        data_pre.append(temp)

    # data_json = merge_to_json(data, cluster, file_name)
    item = {
        'doc_id': file_name,
        'data': data_pre,
        'cluster': cluster
    }
    return item

if __name__ == "__main__":
    # 处理一下对齐的问题：
    # 输入对齐的amr文件，深度优先遍历，便利之后得到对齐下标，然后可以用正常文本得到AMR数据。
    # 写入最终json文件：
    # ::id
    data = []
    xml_file_names = os.listdir(xml_path)
    doc_len = []
    #xml_file_names = [os.listdir(xml_path)[169]]
    for i, xml in enumerate(xml_file_names):
        print(i)
        if i == 4:
            a=1

        file_name = xml.rstrip('.xml')
        parser = ET.parse(xml_path + '/' + file_name + ".xml")
        # a=parser.getroot()[0].attrib['docid']
        data_per_doc, links_per_doc = preprocess(parser, file_name)
        item = pre_to_json(data_per_doc, links_per_doc, file_name)
        doc_len.append([file_name, len(item['data'])])
        data.append(item)

    write_train_file = 'data/corpora-base/train'
    write_dev_file = 'data/corpora-base/dev'
    write_test_file = 'data/corpora-base/test'
    write_evl_file = 'data/corpora-base/evl'

    test_ids = ['007', '027', '041', '063', '077', '081', '093', '095', '134']
    dev_ids = ['006', '029', '042', '064', '078', '082', '094', '096', '135']
    evl_ids = ['009', '028']
    train_data, dev_data, test_data, evl_data = [], [], [], []
    for i, item in enumerate(data):
        if item['doc_id'][-7:-4] == 'dfa':
            if item['doc_id'][-3:] in dev_ids:
                dev_data.append(item)
            elif item['doc_id'][-3:] in test_ids:
                test_data.append(item)
            elif item['doc_id'][-3:] in evl_ids:
                evl_data.append(item)
            else:
                train_data.append(item)
        else:
            train_data.append(item)

    with open(write_train_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(train_data, jsonfile)
    with open(write_evl_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(evl_data, jsonfile)
    with open(write_dev_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(dev_data, jsonfile)
    with open(write_test_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(test_data, jsonfile)

    # data split

    # with open(write_train_file, 'w', encoding='utf-8') as jsonfile:
    #     json.dump(train_data, jsonfile)


    print('done!')






