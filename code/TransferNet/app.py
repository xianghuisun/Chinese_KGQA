import torch
import torch.nn as nn
import math,json
from transformers import AutoModel
import os,re,sys
from tqdm import tqdm
import numpy as np
import time
import traceback
import transformers
from collections import defaultdict
from transformers import AutoTokenizer
from IPython import embed
import logging
from flask import Flask
from flask import request
from utils import convert_tokens_to_ids,toOneHot,paraver,predict_topic_entity,get_answer
from nlp_basictasks.tasks import Ner

####################################### Set logger #######################################
logger=logging.getLogger('main')
logger.setLevel(logging.INFO)
fh=logging.FileHandler('log.txt')
fh.setLevel(logging.INFO)
ch=logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

####################################### Get ent2id,rel2id and all triplets #######################################
kg_folder='/home/xhsun/Desktop/KG/nlpcc2018/knowledge/small_knowledge/'

ent2id = {}
with open(os.path.join(kg_folder, 'entities.dict')) as f:
    lines=f.readlines()
for i in tqdm(range(len(lines))):
    l = lines[i].strip().split('\t')
    ent2id[l[0].strip()] = len(ent2id)
id2ent={k:v for v,k in ent2id.items()}

rel2id = {}
with open(os.path.join(kg_folder, 'relations.dict')) as f:
    lines=f.readlines()
for i in tqdm(range(len(lines))):
    l = lines[i].strip().split('\t')
    rel2id[l[0].strip()] = int(l[1])

triples = []
bad_count=0
with open(os.path.join(kg_folder, 'small_kb')) as f:
    lines=f.readlines()
for i in tqdm(range(len(lines))):
    l = lines[i].strip().split('|||')
    try:
        s = ent2id[l[0].strip()]
        p = rel2id[l[1].strip()]
        o = ent2id[l[2].strip()]
        triples.append((s, p, o))
    except:
        bad_count+=1
        
logger.info('bad count : {}'.format(bad_count))
triples = torch.LongTensor(triples)

####################################### Prepare matrix #######################################
Tsize = len(triples)
Esize = len(ent2id)
num_relations = len(rel2id)

idx = torch.LongTensor([i for i in range(Tsize)])
Msubj = torch.sparse.FloatTensor(
    torch.stack((idx, triples[:,0])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
Mobj = torch.sparse.FloatTensor(
    torch.stack((idx, triples[:,2])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
Mrel = torch.sparse.FloatTensor(
    torch.stack((idx, triples[:,1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, num_relations]))
logger.info('triple size: {}, num_entities : {}, num_relations : {}'.format(Tsize,Esize,num_relations))

####################################### Load KGQA model #######################################
bert_name='/home/xhsun/Desktop/huggingfaceModels/chinese-roberta-wwm/'
tokenizer=AutoTokenizer.from_pretrained(bert_name)

class TransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_steps = 2
        
        self.bert_encoder = AutoModel.from_pretrained(bert_name, return_dict=True)
        dim_hidden = self.bert_encoder.config.hidden_size

        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)

        self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        self.hop_selector = nn.Linear(dim_hidden, self.num_steps)


    def follow(self, e, r):
        x = torch.sparse.mm(Msubj, e.t()) * torch.sparse.mm(Mrel, r.t())
        return torch.sparse.mm(Mobj.t(), x).t() # [bsz, Esize]

    def forward(self, heads, questions):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)

        device = heads.device
        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
            q_dist = q_dist * questions['attention_mask'].float()
            q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
            word_attns.append(q_dist)
            ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]

            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
            # rel_dist = torch.softmax(rel_logit, 1) # bad
            rel_dist = torch.sigmoid(rel_logit)
            rel_probs.append(rel_dist)

            # sub, rel, obj = self.triples[:,0], self.triples[:,1], self.triples[:,2]
            # sub_p = last_e[:, sub] # [bsz, #tri]
            # rel_p = rel_dist[:, rel] # [bsz, #tri]
            # obj_p = sub_p * rel_p
            # last_e = torch.index_add(torch.zeros_like(last_e), 1, obj, obj_p)
            last_e = self.follow(last_e, rel_dist) # faster than index_add

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
        hop_attn = torch.softmax(self.hop_selector(q_embeddings), dim=1).unsqueeze(2) # [bsz, num_hop, 1]
        last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]
        return {
            'e_score': last_e,
            'word_attns': word_attns,
            'rel_probs': rel_probs,
            'ent_probs': ent_probs,
            'hop_attn': hop_attn.squeeze(2)
        }

kgqa_model = TransferNet()
kgqa_model.load_state_dict(torch.load('/home/xhsun/Desktop/code/KG/TransferNet-master/save_dir/original_experiment/model.pt',map_location='cpu'))
logger.info("TransferNet Model has been successfully loaded!!!!")

####################################### Load KGQA model #######################################
ner_model_path='/home/xhsun/Desktop/notebook/TransferNet/NER'
ner_model=Ner(ner_model_path,label2id=None,is_finetune=True,use_crf=True)
logger.info("Ner Model has been successfully loaded!!!!")

vec_len=len(ent2id)

app=Flask(__name__)
@app.route('/kgqa',methods=['POST','GET'])
def kgqa():
    state, conv_request = paraver()
    logger.info('请求参数：{}'.format(json.dumps(conv_request,ensure_ascii=False)))
    #conv_request should be like : {'query':'你知道<香港>有什么明星吗？'}
    result="这个问题暂时无法回答"
    if state == True:
        conv_request = request.json
        start_time = time.time()
        try:
            query =conv_request['query']
        except:
            query = 'error'
            topk = 'error'
            result = error_result
        if query != 'error':
            logger.info("用户问题：{}".format(query))
            try:
                extract_entity=re.findall(pattern='<(.*)>',string=query)
                if extract_entity==[]:
                    topic_entity=predict_topic_entity(ner_model=ner_model,query=query)
                else:
                    topic_entity=extract_entity[0]
                logger.info('topic entity is {}'.format(topic_entity))
                if topic_entity not in ent2id:
                    score=0.0
                    predict_id=-1
                    logger.info("Can not find topic entity {} in entities".format(topic_entity))
                else:
                    score,predict_id=get_answer(model=kgqa_model,query=query,topic_entity=topic_entity,tokenizer=tokenizer,ent2id=ent2id)
                    result=id2ent[predict_id]

            except Exception as error_type:
                abnormal_type = traceback.format_exc()
                log_str = 'get_result-1' + ':' + str(abnormal_type)
                logger.info('error:%s', log_str)
                logger.exception(error_type)

            end_time=time.time()
            spend_time=(end_time-start_time)*1000
            logger.info('整个QA流程总计消耗了{} ms'.format(spend_time))
    else:
        logger.info("state is False")

    logger.info("返回答案：{}".format(result))
    return json.dumps(result,ensure_ascii=False)


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port='12333',use_reloader=False)
