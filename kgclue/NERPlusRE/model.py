from tqdm import tqdm
import json
import random
import os
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from transformers import AutoModel
tokenizer=AutoTokenizer.from_pretrained("/home/xhsun/Desktop/huggingfaceModels/chinese-roberta-wwm/")
from nlp_basictasks.modules.transformers import BertTokenizer,BertModel,BertConfig
from nlp_basictasks.modules import MLP
device=torch.device('cuda')

import logging
logger=logging.getLogger('main.model')

class RelationExtract(nn.Module):
    def __init__(self, model_path,
                num_labels,
                state_dict=None,
                is_finetune=False,
                pooling_type='cls'):
        super().__init__()
        self.num_labels=num_labels
        self.pooling_type=pooling_type
        if is_finetune==False:
            #logger.info("Loading model from {}, which is from huggingface model".format(model_path))
            self.load_huggingface_model(bert_model_path=model_path)
        else:
            self.load_finetuned_model(model_path=model_path)
            #logger.info("Loading model from {}, which has been finetuned.".format(model_path))
            
    def load_huggingface_model(self,bert_model_path):
        self.config=BertConfig.from_pretrained(bert_model_path)
        self.bert=BertModel.from_pretrained(bert_model_path)
        self.head_layer=MLP(in_features=self.config.hidden_size,out_features=self.num_labels)

    def load_finetuned_model(self,model_path):
        bert_save_path=os.path.join(model_path,"BERT")#save的时候将BERT保存在model_path下的BERT文件夹中
        self.config=BertConfig.from_pretrained(bert_save_path)
        self.bert=BertModel.from_pretrained(bert_save_path)

        head_save_path=os.path.join(model_path,'MLP')
        self.head_layer=MLP.load(input_path=head_save_path)

    def save(self,output_path):
        bert_save_path=os.path.join(output_path,"BERT")
        self.bert.save_pretrained(bert_save_path,save_config=False)#下面已经save，不用save两次，虽然没什么影响
        self.config.save_pretrained(bert_save_path)
        head_save_path=os.path.join(output_path,'MLP')
        self.head_layer.save(head_save_path)
        
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,label_ids=None,relation_range=None,output_all_encoded_layers=False):
        '''
        input_ids.size()==attention_mask.size()==token_type_ids.size()==position_ids.size()==
        (batch_size,seq_length)
        label_ids.size()==(batch_size,self.num_labels)
        relation_range.size()==(batch_size,self.num_labels)，是一个one_hot向量，与问题中topic实体相连的关系的位置是1
        '''
        (sequence_outputs,pooled_output)=self.bert(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask,
                                                  output_all_encoded_layers=False)
        #要注意到sequence_output[0]与pooled_output的区别在于pooled_output是经过一层tanh的
        assert len(pooled_output.size())==2 and pooled_output.size(1)==self.config.hidden_size

        if self.pooling_type=='cls':
            before_logits=pooled_output
        elif self.pooling_type=='last_layer':
            #取最后一层的mean pooling
            if output_all_encoded_layers==True:
                assert type(sequence_outputs)==list
                sequence_outputs=sequence_outputs[-1]
            #sequence_outputs现在代表last layer的hidden_states
            attention_mask_expanded=attention_mask.unsqueeze(-1).expand(sequence_outputs.size())
            #sequence_outputs.size()==(bsz,pad_seq_len,dim)==attention_mask_expanded.size()

            sum_mask=attention_mask.sum(1).unsqueeze(1)#(batch_size,1)
            sum_mask=torch.clamp(sum_mask,min=1e-7)
            before_logits=torch.sum(sequence_outputs*attention_mask_expanded,1)/sum_mask#(bsz,seq_len,dim)-->(bsz,dim)
        elif self.pooling_type=='last_two_layer':
            #print(output_all_encoded_layers,type(sequence_outputs))
            assert output_all_encoded_layers==True and type(sequence_outputs)==list
            attention_mask_expanded=attention_mask.unsqueeze(-1).expand(sequence_outputs[-1].size())
            #sequence_outputs.size()==(bsz,pad_seq_len,dim)==attention_mask_expanded.size()
            sum_mask=attention_mask.sum(1).unsqueeze(1)#(batch_size,1)
            sum_mask=torch.clamp(sum_mask,min=1e-7)
            #print(sequence_outputs[-1].size(),attention_mask_expanded.size())
            last_1_pooling=torch.sum(sequence_outputs[-1]*attention_mask_expanded,1)/sum_mask
            last_2_pooling=torch.sum(sequence_outputs[-2]*attention_mask_expanded,1)/sum_mask

            before_logits=(last_1_pooling+last_2_pooling)/2
        
        else:
            raise Exception("Unknown pooling type %s error"%self.pooling_type)
        
        logits=self.head_layer(before_logits)#(batch_size,num_labels)
        if not self.training:
            scores,idx=torch.max(logits,dim=1)#[batch_size],[batch_size]
            return {"predict_scores":scores,"predict_idx":idx}
        else:
            assert label_ids.size()==logits.size()
            if relation_range is not None:
                weight=label_ids*99+1#答案位置的值是100，其余所有位置的值是1
                weight=weight*relation_range#答案位置的值是100，与topic实体相连的所有关系的值是1，其余的关系值是0
            else:
                weight=torch.ones_like(label_ids)
                
            loss=torch.sum(weight*torch.pow(logits-label_ids,2))/torch.sum(weight)
            return {"loss":loss}