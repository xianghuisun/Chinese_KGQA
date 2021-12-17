import os,json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
import time
from transformers import AdamW,AutoTokenizer
import logging
from visdom import Visdom
from nlp_basictasks.modules.transformers import BertTokenizer,BertModel,BertConfig
from nlp_basictasks.modules import MLP

from data import DataLoaderForRE,read_data
from model import RelationExtract
from utils import RAdam,get_linear_schedule_with_warmup

save_dir='../save_dir'
os.makedirs(save_dir,exist_ok=True)

logger=logging.getLogger('main')
logger.setLevel(logging.INFO)

fh=logging.FileHandler(os.path.join(save_dir,'log.txt'))
fh.setLevel(logging.INFO)

ch=logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d : %(message)s')

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
#logger.handlers=[fh]
logger.addHandler(ch)


torch.set_num_threads(1) # avoid using multiple cpus

vis=Visdom(env='main',log_to_filename=os.path.join(save_dir,'vis_log'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):

    with open(args.rel2id_dir) as f:
        rel2id=json.load(f)
    tokenizer=AutoTokenizer.from_pretrained(args.bert_name)

    train_examples,test_examples=read_data(args.data_dir)

    train_loader=DataLoaderForRE(all_examples=train_examples,label2id=rel2id,tokenizer=tokenizer,batch_size=args.batch_size,shuffle=True)
    test_loader=DataLoaderForRE(all_examples=test_examples,label2id=rel2id,tokenizer=tokenizer,batch_size=args.batch_size,shuffle=False)

    print_loss_step=len(train_loader)//5
    evaluation_steps=len(train_loader)//2

    model=RelationExtract(model_path=args.bert_name,num_labels=len(rel2id),pooling_type='last_layer')
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)

    t_total = len(train_loader) * args.num_epoch
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0, 'lr': args.bert_lr},
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0, 'lr': args.lr},
        ]
    # optimizer_grouped_parameters = [{'params':model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}]
    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optimizer_grouped_parameters)
    else:
        raise NotImplementedError
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # validate(args, model, val_loader, device)
    logger.info("Start training........")
    previous_acc=0.0
    model.eval()
    acc = predict(model=model, test_loader=test_loader)
    logger.info('Accuracy before training is {}'.format(acc))
    global_step=0
    for epoch in tqdm(range(args.num_epoch)):
        model.zero_grad()
        model.train()
        training_loss=0.0
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            question_features,relation_id_tensor,relation_range_tensor=batch

            input_features={
                'input_ids':question_features['input_ids'],
                'attention_mask':question_features['attention_mask'],
                'token_type_ids':question_features['token_type_ids'],
                'label_ids':relation_id_tensor,
                'relation_range':relation_range_tensor
            }
            for key,value in input_features.items():
                input_features[key]=value.to(device)

            loss = model(**input_features)['loss']
            training_loss+=loss.item()

            vis.line([optimizer.param_groups[0]['lr']],[global_step],win=f'learning rate',
                    opts=dict(title=f'learning rate', xlabel='step', ylabel='lr'), update='append')
            vis.line([loss.item()],[global_step],win=f'training loss',
                    opts=dict(title=f'training loss', xlabel='step', ylabel='loss'), update='append')
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step+=1
            optimizer.zero_grad()

            if iteration % print_loss_step == 0:
                training_loss/=print_loss_step
                logger.info("Epoch : {}, train_step : {}/{}, loss_value : {} ".format(epoch,iteration*(epoch+1),t_total,training_loss))
                training_loss=0.0

            if iteration % evaluation_steps == 0:
                model.eval()
                acc=predict(model=model,test_loader=test_loader)
                logger.info("In epoch {}, accuracy is {}".format(epoch+1,acc))
                vis.line([acc],[global_step],win=f'accuracy',
                        opts=dict(title=f'accuracy', xlabel='step', ylabel='accuracy', markers=True, markersize=10), update='append')
                if acc> previous_acc:
                    model.save(output_path=save_dir)
                    logger.info("previous acc is {} and current acc is {}".format(previous_acc,acc))
                    previous_acc=acc
                model.zero_grad()
                model.train()

def predict(model,test_loader):
    for iteration, batch in enumerate(test_loader):
        iteration = iteration + 1
        question_features,relation_id_tensor,relation_range_tensor=batch
        input_features={
            'input_ids':question_features['input_ids'],
            'attention_mask':question_features['attention_mask'],
            'token_type_ids':question_features['token_type_ids'],
            'label_ids':relation_id_tensor,
            'relation_range':relation_range_tensor
        }
        for key,value in input_features.items():
            input_features[key]=value.to(device)

        result = model(**input_features)
        predict_ids=result['predict_idx']#(batch_size,)
        label_ids=torch.argmax(relation_id_tensor,dim=1).to(device)#(batch_size,)
        assert predict_ids.size()==label_ids.size()
        return (predict_ids==label_ids).float().mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = './', help='path to the data')
    parser.add_argument("--rel2id_dir",default = './rel2id.json')
    parser.add_argument('--save_dir', default=save_dir, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--max_grad_norm', default=1, type = int)
    # model parameters
    parser.add_argument('--bert_name', default='/home/xhsun/Desktop/huggingfaceModels/chinese-roberta-wwm')
    args = parser.parse_args()

    # args display
    args_config={}
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))
        args_config[k]=v

    with open(os.path.join(save_dir,'args_config.dict'),'w') as f:
        json.dump(args_config,f,ensure_ascii=False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()