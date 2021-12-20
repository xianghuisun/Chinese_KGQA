from collections import defaultdict
from tqdm import tqdm
import re
import json
from copy import deepcopy
import random
import torch
import numpy as np
from nlp_basictasks.tasks import Ner,cls
from nlp_basictasks.evaluation import nerEvaluator,clsEvaluator
from collections import Counter
from copy import deepcopy
from nlp_basictasks.modules.transformers import BertTokenizer
device=torch.device('cuda')
from nlp_basictasks.tasks import cls
from nlp_basictasks.evaluation import pairclsEvaluator
from nlp_basictasks.readers.paircls import InputExample

def read_data(data_path):
    with open(data_path) as f:
        lines=f.readlines()
    data=[]
    for line in lines:
        data.append(json.loads(line))
    print("原始数据有{}个样本".format(len(data)))
    return data

#第一步：预测测试集中的实体########################################################################
data=read_data('/home/xhsun/Desktop/KG/kgCLUE/qa_data/test.json')

print(data[0])

########################################################加载NER模型
ner_model_path='../NER/'
ner_model=Ner(ner_model_path,label2id=None,is_finetune=True,use_crf=True)
ner_model._target_device='cuda'

########################################################利用NER模型预测测试句子中的实体
def predict_topic_entity(ner_model,query):
    predict_label_list=ner_model.predict(query)[0]
    sentence_list=ner_model.model.tokenizer.tokenize(query)
    assert len(predict_label_list)==len(sentence_list)
    topic_entity=''
    for i in range(len(sentence_list)):
        token=sentence_list[i]
        predict_label=predict_label_list[i]
        if predict_label!='O':
            topic_entity+=token
    return topic_entity

all_questions=[example['question'] for example in data]
all_predict_heads=ner_model.predict(all_questions)

print(all_questions[0],all_predict_heads[0])
for i in range(len(data)):
    predict_label_list=all_predict_heads[i]
    sentence_list_by_tokenizer=ner_model.model.tokenizer.tokenize(data[i]['question'])
    sentence_list=list(data[i]['question'])
    if len(sentence_list_by_tokenizer)!=len(sentence_list):
        assert len(sentence_list)>len(sentence_list_by_tokenizer)
        sentence_list=sentence_list_by_tokenizer
    assert len(predict_label_list)==len(sentence_list)
    topic_entity=''
    for k in range(len(sentence_list)):
        token=sentence_list[k]
        predict_label=predict_label_list[k]
        if predict_label!='O':
            topic_entity+=token
    topic_entity=topic_entity.replace('##','')
    data[i].update({'topic_entity':topic_entity})

########################################################预测的实体中有一些不准确的case，通过一些规则提升
def ParsingEntity(question,topic_entity):
    try:
        assert len(topic_entity)>=5
        start_token=topic_entity[:2].lower()#前两个字符
        end_token=topic_entity[-2:].lower()#后两个字符
        copy_question=question.lower()
        pattern=start_token+'(.*)'+end_token
        result=re.search(pattern=pattern,string=copy_question)
        start_id,end_id=result.span()
        return question[start_id:end_id]
    except:
        return ''
    
count=0
for i in range(len(data)):
    question,topic_entity=data[i]['question'],data[i]['topic_entity']
    if topic_entity not in question:
        if topic_entity.upper() in question:
            data[i]['topic_entity']=topic_entity.upper()
        elif topic_entity[0].upper()+topic_entity[1:] in question:
            data[i]['topic_entity']=topic_entity[0].upper()+topic_entity[1:]
        elif ParsingEntity(question,topic_entity=topic_entity)!='':
            data[i]['topic_entity']=ParsingEntity(question,topic_entity=topic_entity)
        else:
            count+=1
            print(data[i])

#################################################第一步，预测实体的步骤结束#####################################

#第二步：根据实体找出所有候选的三元组，并且与问题计算相似度##########################################################################################

###########################################################读取别名字典和sub_map字典，从而根据实体，找出对应的别名实体，同时找出所有候选三元组
with open('/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/alias_dict.json') as f:
    alias_dict=json.load(f)
    
with open('/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/sub_map.json') as f:
    sub_map=json.load(f)

for i in range(len(data)):
    example=data[i]
    question,topic_entity=example['question'],example['topic_entity']
    alias_entities=[]
    if topic_entity in alias_dict:
        alias_entities=alias_dict[topic_entity]
    data[i].update({"alias_entities":alias_entities})
    candidate_triples=set()
    for alias_entity in alias_entities:
        for p,o in sub_map[alias_entity]:
            candidate_triples.add((alias_entity,p,o))
    if topic_entity in sub_map:
        for p,o in sub_map[topic_entity]:
            candidate_triples.add((topic_entity,p,o))
    candidate_triples=list(candidate_triples)
    data[i].update({"candidate_triples":candidate_triples})

###########################################################加载匹配模型
paircls_model=cls(model_path="/home/xhsun/NLP/KGQA/KG/kgCLUE/NER-Match/Match/",is_finetune=True,device='cuda')

def predict_match_relation(paircls_model,question_triple_pairs):
    number_of_triplets=len(question_triple_pairs)
    predict_probs=paircls_model.predict(is_pairs=True,dataloader=question_triple_pairs)
    assert predict_probs.shape==(number_of_triplets,2)
    predict_id=np.argmax(predict_probs[:,1])
    return predict_id

topic_entity_not_found_num=0
text_match_relation_num=0
model_predict_num=0
for i in tqdm(range(len(data))):
    example=data[i]
    question,topic_entity,candidate_triples=example['question'],example['topic_entity'],example['candidate_triples']
    assert topic_entity in question
    question=question.replace(topic_entity,'NE')
    
    question_triple_pairs=[]
    
    final_triple=None
    for head,relation,tail in candidate_triples:
        question_triple_pairs.append([question,','.join([head,relation,tail])])
        if relation in question:
            final_triple=[head,relation,tail]
            text_match_relation_num+=1
            break
            
    if final_triple==None:
        assert len(question_triple_pairs)==len(candidate_triples)
        if len(candidate_triples)==0:
            assert topic_entity not in sub_map
            topic_entity_not_found_num+=1
            final_triple=[topic_entity,topic_entity,topic_entity]
        else:
            predict_id=predict_match_relation(paircls_model=paircls_model,question_triple_pairs=question_triple_pairs)
            final_triple=candidate_triples[predict_id]
            model_predict_num+=1
    data[i].update({"answer":str(final_triple[0])+" ||| "+str(final_triple[1])+" ||| "+str(final_triple[2])})

print(model_predict_num,topic_entity_not_found_num,text_match_relation_num)
###################################################至此，第二步匹配阶段结束

#####################################################提交样例
submit_examples=[]
for example in data:
    id_,ans=example['id'],example['answer']
    submit_examples.append({"id":id_,"answer":ans})

with open("/home/xhsun/Desktop/kgclue_predict.json","w") as f:
    for predict_example in submit_examples:
        f.write(json.dumps(predict_example,ensure_ascii=False)+'\n')


