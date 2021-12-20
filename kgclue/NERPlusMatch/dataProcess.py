from collections import defaultdict
from tqdm import tqdm
import re
import json
from copy import deepcopy
import random

import torch
import os
import pickle
from transformers import AutoTokenizer
import transformers
import torch.nn as nn
import math
import time
import pandas as pd
import numpy as np
from nlp_basictasks.tasks import Ner,cls
from nlp_basictasks.evaluation import nerEvaluator,clsEvaluator
from nlp_basictasks.readers.cls import InputExample
from collections import Counter
from copy import deepcopy
from nlp_basictasks.modules.transformers import BertTokenizer
device=torch.device('cuda')

def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line))
    print("原始数据有{}个样本".format(len(data)))
    return data
    
data=[]
data.extend(read_data('/home/xhsun/NLP/KGQA/QApairs/Chinese/kgclue/train.json'))
data.extend(read_data('/home/xhsun/NLP/KGQA/QApairs/Chinese/kgclue/dev.json'))
data.extend(read_data('/home/xhsun/NLP/KGQA/QApairs/Chinese/kgclue/test_public.json'))

sub_map = defaultdict(set)
obj_map = defaultdict(set)
so_map=defaultdict(set)
sp_map=defaultdict(set)
alias_dict=defaultdict(set)
kg_path='/home/xhsun/NLP/KGQA/KG/kgCLUE/Knowledge.txt'
with open(kg_path) as f:
    lines=f.readlines()
    
entities_set=set()
relations_set=set()
print(len(lines))

#new_triples=[]
for i in tqdm(range(len(lines))):
    line=lines[i]
    l = line.strip().split('\t')
    s = l[0].strip()
    p = l[1].strip()
    o = l[2].strip()
    if s=='' or p=='' or o=='':
        print('-'*100)
        continue
    entities_set.add(s)
    entities_set.add(o)
    relations_set.add(p)
    #new_triples.append((s,p,o))
    sub_map[s].add((p, o))
    obj_map[o].add((s, p))
    so_map[(s,o)].add(p)
    sp_map[(s,p)].add(o)
    if '（' in s and "）" in s:
        entity_mention=s.split('（')[0]
        alias_dict[entity_mention].add(s)

print('大规模图谱中一共有{}个三元组，{}个实体，{}个关系'.format(len(lines),len(entities_set),len(relations_set)))

relation_match_data=[]
pos_nums=0
neg_nums=0
for example in tqdm(data):
    id_,question,answer=example['id'],example['question'],example['answer']
    answer_split=answer.split('|||')
    head,relation,tail=answer_split[0].strip(),answer_split[1].strip(),answer_split[2].strip()
    entity_mention=head
    if ('（' in head and '）' in head) and head not in question:
        entity_mention=head.split('（')[0]
        assert entity_mention in alias_dict
    if relation in ['别名','中文名',"英文名","外文名","原名","别称","昵称","全称","中文名称","英文名称"]:
        alias_dict[entity_mention].add(tail)
        
    candidate_pos=sub_map[head]
#     print(relation,candidate_relations)
#     assert relation in candidate_relations
    if entity_mention not in question:
        continue
    question=question.replace(entity_mention,'NE')
    relation_match_data.append({"question":question,"head":head,'relation':relation,'label':'1',"answer":tail})
    pos_nums+=1
    for p,o in candidate_pos:
        if p!=relation:
            relation_match_data.append({"question":question,"head":head,'relation':p,'label':'0','answer':o})               
            neg_nums+=1

print(len(relation_match_data),pos_nums,neg_nums)

with open("/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/sub_map.json","w") as f:
    for key,value in sub_map.items():
        sub_map[key]=list(value)
    json.dump(obj=sub_map,fp=f,ensure_ascii=False)
with open("/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/alias_dict.json","w") as f:
    for key,value in alias_dict.items():
        alias_dict[key]=list(value)
    json.dump(obj=alias_dict,fp=f,ensure_ascii=False)
with open("/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/match_data.txt","w") as f:
    for example in relation_match_data:
        f.write(json.dumps(example,ensure_ascii=False)+'\n')