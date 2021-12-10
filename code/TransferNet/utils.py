import torch
import math
import numpy as np
import warnings
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import transformers
from collections import defaultdict, Counter, deque
import json
import pickle

from flask import request
import traceback
import logging
logger=logging.getLogger("main.utils")

def batch_device(batch, device):
    res = []
    for x in batch:
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        elif isinstance(x, (dict, transformers.tokenization_utils_base.BatchEncoding)):
            for k in x:
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].to(device)
        elif isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
            x = list(map(lambda i: i.to(device), x))
        res.append(x)
    return res


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

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss