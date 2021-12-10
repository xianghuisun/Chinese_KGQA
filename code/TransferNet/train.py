import os,json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np
import time
from .utils import MetricLogger, batch_device, RAdam, get_linear_schedule_with_warmup
from .data import load_data
from .model import TransferNet
from .predict import validate
from transformers import AdamW
import logging
from visdom import Visdom

save_dir='./save_dir'
os.makedirs(save_dir,exist_ok=True)

logger=logging.getLogger('train')
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

vis=Visdom(env='train',log_to_filename=os.path.join(save_dir,'vis_log'))

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ent2id, rel2id, triples, train_loader, val_loader = load_data(args.knowledge_dir,args.qa_dir, args.bert_name, args.batch_size, args.cache_dir)
    logging.info("Create model.........")
    logging.info("Entity nums : {} and relation nums : {}, triple nums : {}".format(len(ent2id),len(rel2id),len(triples)))
    model = TransferNet(args, ent2id, rel2id, triples)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)

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
    meters = MetricLogger(delimiter="  ")
    # validate(args, model, val_loader, device)
    logger.info("Start training........")
    previous_acc=0.0
    acc = validate(args, model, val_loader, device)
    logger.info('Accuracy before training is {}'.format(acc))
    global_step=0
    for epoch in tqdm(range(args.num_epoch)):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            loss = model(*batch_device(batch, device))

            vis.line([optimizer.param_groups[0]['lr']],[global_step],win=f'learning rate',
                    opts=dict(title=f'learning rate', xlabel='step', ylabel='lr'), update='append')
            vis.line([loss['loss'].item()],[global_step],win=f'training loss',
                    opts=dict(title=f'training loss', xlabel='step', ylabel='loss'), update='append')
            
            optimizer.zero_grad()
            if isinstance(loss, dict):
                if len(loss) > 1:
                    total_loss = sum(loss.values())
                else:
                    total_loss = loss[list(loss.keys())[0]]
                meters.update(**{k:v.item() for k,v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()
            global_step+=1

            if iteration % (len(train_loader) // 5) == 0:
            # if True:
                
                logger.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )

        acc = validate(args, model, val_loader, device)
        logger.info("In epoch {}, accuracy is {}".format(epoch+1,acc))
        vis.line([acc],[global_step],win=f'accuracy',
                opts=dict(title=f'accuracy', xlabel='step', ylabel='accuracy', markers=True, markersize=10), update='append')
        
        if acc> previous_acc:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
            logger.info("previous acc is {} and current acc is {}".format(previous_acc,acc))
            previous_acc=acc

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--knowledge_dir', default = '/home/xhsun/Desktop/KG/nlpcc2018/knowledge/small_knowledge', help='path to the KG')
    parser.add_argument('--qa_dir', default = '/home/xhsun/Desktop/KG/QApairs/ChineseQA', help='path to the data')
    parser.add_argument('--cache_dir', default = './cache_dir/Chinese', help='path to the data')
    parser.add_argument('--save_dir', default=save_dir, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
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
