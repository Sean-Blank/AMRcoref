import torch
import re
import string

print(torch.cuda.is_available())
from xml.etree import ElementTree as ET
import os
from amr_parsing.io import AMRIO
from operator import itemgetter, attrgetter


def get_all_multi_ids(path):
    files = os.listdir(path)
    ids_info = []
    ids = []
    for file in files:
        parser = ET.parse(path + '/' + file)
        id_per_doc = []
        for i, inst in enumerate(parser.getroot()[0]):
            ids.append(inst.get('id').rstrip(string.digits).rstrip("."))
            id_per_doc.append([inst.get('id'), inst.get('speaker'), int(inst.get('order')),
                               inst.get('post'), file[6:9]])
        sorted(id_per_doc, key=itemgetter(2))
        ids_info.append(id_per_doc)
    ids = set(ids)
    return ids_info, ids


# step1: merge and extract multi-sentence amr raw text from amr_3.0 unsplit data.
# os.system('awk FNR!=1 data/amrs/unsplit/* > all.txt')
raw_path = '../data/amr3.txt'
xml_path = '../data/xml-unsplit/'
txt_path = '../data/amr-unsplit'
ms_amr_id_info, ms_amr_ids = get_all_multi_ids(xml_path)
align_path = '../data/amr-alignments.txt'
# step2: get the filtered amr and write

ms_amr = []
with open('./data/amr3_filtered', 'w', encoding='utf-8') as f:
    for i, amr in enumerate(AMRIO.read(raw_path)):
        print(i)
        current_id = amr.id.split("::date", 1)[0].strip().rstrip(string.digits).rstrip(".")
        if current_id in ms_amr_ids:
            print(i)
            ms_amr.append(amr)
            AMRIO.dump([amr], f)


xml_files = os.listdir(xml_path)


def get_new_ids(id_info):
    new_ids_info, new_ids = [], []
    for i, line in enumerate(id_info):
        new_ids_info.append(line[0] + " ::speaker "
                             + line[1] + " ::order " + str(line[2])
                            +" ::post " + line[3] + " ::doc_type " + line[4])
        new_ids.append(line[0])
    return new_ids, new_ids_info

# step3: from the filtered amr text, generate the single doc text
for i, (id_info, file) in enumerate(zip(ms_amr_id_info, xml_files)):
    with open(txt_path + '/' + file.rstrip('.xml') + '.txt', 'w', encoding='utf-8') as m:
        m.write('\n')
        ids, new_ids = get_new_ids(id_info)

        for k, amr in enumerate(AMRIO.read('../data/amr3_filtered')):
            for j, (id, new_id) in enumerate(zip(ids, new_ids)):
                if id == amr.id.split("::date", 1)[0].strip():
                    amr.id = new_id
                    print(i, j)
                    AMRIO.dump([amr], m)
                    break



