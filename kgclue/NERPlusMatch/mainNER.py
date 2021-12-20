from collections import defaultdict
from tqdm import tqdm
import re
import json
from copy import deepcopy
import random
import torch
import sys,os
import pandas as pd
import numpy as np
from nlp_basictasks.tasks import Ner
from nlp_basictasks.evaluation import nerEvaluator
from nlp_basictasks.readers.ner import InputExample
from collections import Counter
from copy import deepcopy
from nlp_basictasks.modules.transformers import BertTokenizer
tokenizer=BertTokenizer.from_pretrained('/home/xhsun/Desktop/huggingfaceModels/chinese-roberta-wwm/')

def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line))
    print("原始数据有{}个样本".format(len(data)))
    return data
    
data=[]
data.extend(read_data('/home/xhsun/Desktop/KG/kgCLUE/KgCLUE-main/qa_data/train.json'))
data.extend(read_data('/home/xhsun/Desktop/KG/kgCLUE/KgCLUE-main/qa_data/test_public.json'))
data.extend(read_data('/home/xhsun/Desktop/KG/kgCLUE/KgCLUE-main/qa_data/dev.json'))


labels=[]
sentences=[]
examples=[]
found_num=0
for example in data:
    question,answer=example['question'],example['answer']
    triplet=answer.strip().split('|||')
    assert len(triplet)==3
    entity=triplet[0]
    assert type(question)==str
    question=question.strip()
    found=False
    positions=[]
    entity=entity.strip()
    
    if '（' in entity and entity.endswith('）'):
        entity=entity.split('（')[0]
        
    if entity in question:
        start_id=question.find(entity)
        end_id=start_id+len(entity)
        positions.append((start_id,end_id))
        found=True
        
    if found:
        found_num+=1
        sentences.append(question)
        label=['O']*len(question)
        for each_span in positions:
            start_id,end_id=each_span
            for i in range(start_id,end_id):
                if i==start_id:
                    label[i]='B-ent'
                else:
                    label[i]='I-ent'
        labels.append(label)
        examples.append(example)


print(found_num,len(data))

print(len(sentences),len(labels))
for i in range(10):
    print(sentences[i],labels[i],examples[i])
    print('-'*100)


index=list(range(len(sentences)))
random.shuffle(index)
test_nums=int(len(index)*0.1)
print(test_nums)
train_sentences=[]
train_labels=[]
train_raws=[]
for i in range(len(index)-test_nums):
    train_sentences.append(list(sentences[index[i]]))
    train_labels.append(labels[index[i]])
    train_raws.append(examples[index[i]])
    
k=10
print(train_sentences[k],train_labels[k],train_raws[k])

test_sentences=[]
test_labels=[]
test_raws=[]
for i in range(len(index)-test_nums,len(index)):
    test_sentences.append(list(sentences[index[i]]))
    test_labels.append(labels[index[i]])
    test_raws.append(examples[index[i]])
    
print(test_sentences[k],test_labels[k],test_raws[k])

train_examples=[]
for seq_in,seq_out in zip(train_sentences,train_labels):
    train_examples.append(InputExample(seq_in=seq_in,seq_out=seq_out))

test_examples=[]
for seq_in,seq_out in zip(test_sentences,test_labels):
    test_examples.append(InputExample(seq_in=seq_in,seq_out=seq_out))

print(len(train_examples),len(test_examples))
label2id={'[PAD]':0,'O':1,'B-ent':2,'I-ent':3}

model_path='/home/xhsun/Desktop/huggingfaceModels/chinese-roberta-wwm/'
output_path='../NER'
ner_model=Ner(model_path,label2id=label2id,use_crf=True,use_bilstm=False,device='cuda',batch_first=True,
              tensorboard_logdir='../NER/log.txt')

from torch.utils.data import DataLoader
batch_size=64
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

evaluator=nerEvaluator(label2id=label2id,seq_in=test_sentences,seq_out=test_labels,write_csv=True)

ner_model.fit(train_dataloader=train_dataloader,evaluator=evaluator,epochs=5,output_path=output_path)
