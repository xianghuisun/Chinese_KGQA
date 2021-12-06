import torch
import os,re
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from utils.misc import invert_dict
from tqdm import tqdm

def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, entity_range = batch
    topic_entity = torch.stack(topic_entity)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = torch.stack(answer)
    entity_range = torch.stack(entity_range)
    return topic_entity, question, answer, entity_range


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id):
        self.questions = questions
        self.ent2id = ent2id

    def __getitem__(self, index):
        topic_entity, question, answer, entity_range = self.questions[index]
        topic_entity = self.toOneHot(topic_entity)
        answer = self.toOneHot(answer)
        entity_range = self.toOneHot(entity_range)
        return topic_entity, question, answer, entity_range

    def __len__(self):
        return len(self.questions)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.ent2id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, kg_path, qa_file, bert_name, ent2id, rel2id, batch_size, training=False):
        print('Reading questions from {}'.format(qa_file))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)



        sub_map = defaultdict(list)
        so_map = defaultdict(list)
        with open(kg_path) as f:
            lines=f.readlines()
        for i in tqdm(range(len(lines))):
            line=lines[i]
            l = line.strip().split('|||')
            s = l[0].strip()
            p = l[1].strip()
            o = l[2].strip()
            sub_map[s].append((p, o))
            so_map[(s, o)].append(p)


        data = []
        with open(qa_file) as f:
            lines=f.readlines()
        for i in tqdm(range(len(lines))):
            line=lines[i]
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            # if no answer
            if len(line) != 2:
                continue
            # question = line[0].split('[')
            # question_1 = question[0]
            # question_2 = question[1].split(']')
            # head = question_2[0].strip()
            # question_2 = question_2[1]
            # # question = question_1 + 'NE' + question_2
            # question = question_1.strip()
            ans = line[1].strip().split('|')
            question=line[0]
            topic_entity=re.findall('<(.*)>',question)[0]
            head=topic_entity
            question=question.replace('<'+topic_entity+'>','NE')

            # if (head, ans[0]) not in so_map:
            #     continue

            entity_range = set()
            for p, o in sub_map[head]:
                entity_range.add(o)
                for p2, o2 in sub_map[o]:
                    entity_range.add(o2)
            entity_range = [ent2id[o] for o in entity_range]

            head = [ent2id[head]]
            question = self.tokenizer(question.strip(), max_length=64, padding='max_length', return_tensors="pt")
            ans = [ent2id[a.strip()] for a in ans]
            data.append([head, question, ans, entity_range])

        print('data number: {}'.format(len(data)))
        
        dataset = Dataset(data, ent2id)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )


def load_data(kg_folder,qas_dir, bert_name, batch_size, cache_dir):
    #/home/xhsun/Desktop/KG/nlpcc2018/knowledge
    print('Read data...')
    ent2id = {}
    with open(os.path.join(kg_folder, 'entities.dict')) as f:
        lines=f.readlines()
    for i in tqdm(range(len(lines))):
        l = lines[i].strip().split('\t')
        ent2id[l[0].strip()] = len(ent2id)
    # print(len(ent2id))
    # print(max(ent2id.values()))
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
        # p_rev = rel2id[l[1].strip()+'_reverse']
        # triples.append((o, p_rev, s))
    print('bad count : ',bad_count)
    triples = torch.LongTensor(triples)

    train_data = DataLoader(os.path.join(kg_folder, 'small_kb'), 
                            os.path.join(qas_dir, 'train.txt'), bert_name, ent2id, rel2id, batch_size, training=True)
    test_data = DataLoader(os.path.join(kg_folder, 'small_kb'), 
                            os.path.join(qas_dir, 'test.txt'), bert_name, ent2id, rel2id, batch_size)

    return ent2id, rel2id, triples, train_data, test_data
