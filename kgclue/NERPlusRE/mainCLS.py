import sys,os
import pandas as pd
import random
import json
import numpy as np
from torch.utils.data import DataLoader
from nlp_basictasks.tasks import cls
from nlp_basictasks.evaluation import clsEvaluator
from nlp_basictasks.readers.cls import getExamplesFromData

device='cuda'
model_path='/home/xhsun/Desktop/huggingfaceModels/chinese-roberta-wwm/'
output_path="../save_dir/"#output_path是指保存模型的路径
tensorboard_logdir='../save_dir/log'
batch_size=64
optimizer_params={'lr':5e-5}
epochs=10
rel2id_dir='./rel2id.json'

with open(rel2id_dir) as f:
    label2id=json.load(f)

def read_data(data_dir):
    def fun(file_name):
        examples=[]
        with open(file_name) as f:
            lines=f.readlines()
            for line in lines:
                examples.append(json.loads(line))
        return examples
    train_examples=fun(os.path.join(data_dir,'train_data.json'))
    test_examples=fun(os.path.join(data_dir,'test_data.json'))
    return train_examples,test_examples

train_examples,dev_examples=read_data('./')
train_sentences=[example['question'] for example in train_examples]
train_labels=[example['relation'] for example in train_examples]

dev_sentences=[example['question'] for example in dev_examples]
dev_labels=[example['relation'] for example in dev_examples]

print("训练数据个数：",len(train_sentences),"测试数据个数：",len(dev_sentences))

train_examples,max_seq_len=getExamplesFromData(sentences=train_sentences,labels=train_labels,label2id=label2id,mode='train',return_max_len=True)
dev_examples=getExamplesFromData(sentences=dev_sentences,labels=dev_labels,label2id=label2id,mode='dev')

max_seq_len=min(64,max_seq_len)
print('数据集中最长的句子长度 : ',max_seq_len)
cls_model=cls(model_path=model_path,
                label2id=label2id,
                max_seq_length=max_seq_len,
                device=device,
                tensorboard_logdir=tensorboard_logdir)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

evaluator=clsEvaluator(sentences=dev_sentences,label_ids=dev_labels,write_csv=False,label2id=label2id)

cls_model.fit(is_pairs=False,
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            output_path=output_path,
            epochs=epochs,
            optimizer_params=optimizer_params)