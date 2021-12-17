import torch
import os
import logging
import json

logger=logging.getLogger('main.data')

def collate(batch):
    batch = list(zip(*batch))
    question_features,relation_id_tensor,relation_range_tensor = batch
    relation_id_tensor=torch.stack(relation_id_tensor)#(batch_size,num_relations)
    question_features = {k:torch.cat([q[k] for q in question_features], dim=0) for k in question_features[0]}
    relation_range_tensor = torch.stack(relation_range_tensor)
    return question_features,relation_id_tensor,relation_range_tensor

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, rel2id):
        self.data = data
        self.rel2id = rel2id

    def __getitem__(self, index):
        question_features,relation_id,relation_range = self.data[index]
        relation_id_tensor = self.toOneHot(relation_id)#[num_relations]
        relation_range_tensor = self.toOneHot(relation_range)#[num_relations]
        return question_features,relation_id_tensor,relation_range_tensor

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.rel2id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot
    
class DataLoaderForRE(torch.utils.data.DataLoader):
    def __init__(self,all_examples,label2id,tokenizer,batch_size,shuffle=False,max_length=64):
        self.tokenizer=tokenizer
        self.label2id=label2id
        
        data=[]
        for example in all_examples:
            question,relation,other_relations=example['question'],example['relation'],example['other_relations']
            relation_id=[label2id[relation]]
            relation_range=[label2id[rel] for rel in other_relations]
            question_features=self.tokenizer(question.strip(),max_length=max_length,padding="max_length",return_tensors='pt')
            data.append([question_features,relation_id,relation_range])
        print('data number: {}'.format(len(data)))
        
        dataset=Dataset(data=data,rel2id=label2id)
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate, 
            )

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