import torch
from flask import request
import traceback
import logging
logger=logging.getLogger("main.utils")

def convert_tokens_to_ids(tokenizer,ent2id,topic_entity,question):
    if '<'+topic_entity+'>' not in question:
        question=question.replace(topic_entity,'NE')
    else:
        question=question.replace('<'+topic_entity+'>','NE')

    logger.info('Replace topic entity {} by NE : {}'.format(topic_entity,question))
    head=[ent2id[topic_entity]]
    token_ids=tokenizer(question.strip(), max_length=64, padding='max_length', return_tensors="pt")
    return head,token_ids

def toOneHot(indices,vec_len):
    indices = torch.LongTensor(indices)
    one_hot = torch.FloatTensor(vec_len)
    one_hot.zero_()
    one_hot.scatter_(0, indices, 1)
    return one_hot

def paraver():
    state = False
    try:
        state, conv_request = True, request.get_json(force=True)
    except Exception as error_type:
        abnormal_type = traceback.format_exc()
        state,conv_request = False,{'abnormal_type': abnormal_type}
    return state, conv_request

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

def get_answer(model,query,topic_entity,tokenizer,ent2id):
    head,token_ids=convert_tokens_to_ids(tokenizer,ent2id=ent2id,topic_entity=topic_entity,question=query)
    one_hot_head=toOneHot(head,vec_len=len(ent2id))
    with torch.no_grad():
        result=model(*(one_hot_head.unsqueeze(0),token_ids))
    e_score=result['e_score']
    scores,idx=torch.max(e_score,dim=1)
    score=scores.tolist()[0]
    predict_id=idx.tolist()[0]
    return score,predict_id