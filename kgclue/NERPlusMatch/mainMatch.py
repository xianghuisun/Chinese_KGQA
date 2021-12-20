import sys,os
import pandas as pd
import random
import json
import numpy as np
from torch.utils.data import DataLoader
from nlp_basictasks.tasks import cls
from nlp_basictasks.evaluation import pairclsEvaluator
from nlp_basictasks.readers.paircls import InputExample
import logging
logger=logging.getLogger('main')
logger.setLevel(logging.INFO)

device='cuda'
model_path='/home/xhsun/NLP/huggingfaceModels/chinese-roberta-wwm/'
output_path="/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/Match"#output_path是指保存模型的路径
tensorboard_logdir='/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/Match/log'
file_path='/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/match_data.txt'
batch_size=64
optimizer_params={'lr':5e-5}
epochs=5
label2id={"0":0,"1":1}


fh=logging.FileHandler(os.path.join(output_path,'log.txt'))
fh.setLevel(logging.INFO)

ch=logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
#logger.handlers=[fh]
logger.addHandler(ch)

def getExamples(file_path,label2id):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    train_data=[]

    for line in lines:
        example=json.loads(line)

        question=example['question']
        head=example['head']
        relation=example['relation']
        label=example['label']
        answer=example['answer']

        sentence1 = question
        sentence2=str(head)+','+str(relation)+','+str(answer)

        train_data.append(InputExample(text_list=[sentence1,sentence2],label=label2id[str(label)]))
    for _ in range(5):
        i=random.randint(0,len(train_data)-1)
        logger.info("\t".join(train_data[i].text_list)+"\t"+str(train_data[i].label))
    return train_data

all_examples=getExamples(file_path=file_path,label2id=label2id)
random.shuffle(all_examples)
train_examples=all_examples[:-10000]
dev_examples=all_examples[-10000:]

print("Length of train examples : {}".format(len(train_examples)))
batch_size=64
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
dev_sentences=[example.text_list for example in dev_examples]
dev_labels=[example.label for example in dev_examples]
print(dev_sentences[0],dev_labels[0])
evaluator=pairclsEvaluator(sentences_list=dev_sentences,labels=dev_labels,write_csv=True)#定义evaluator


paircls_model=cls(model_path=model_path,label2id=label2id,is_finetune=False,device=device)
paircls_model.fit(is_pairs=True,train_dataloader=train_dataloader,evaluator=evaluator,epochs=epochs,
                  output_path=output_path)