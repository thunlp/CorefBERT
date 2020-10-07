import random, os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from transformers import BertTokenizer, BertForSequenceEncoder, RobertaTokenizer, RobertaForSequenceEncoder
from models import inference_model
from data_loader import DataLoader
from torch.nn import NLLLoss
import logging
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for index, data in enumerate(validset_reader):
        inputs, lab_tensor = data
        prob = model(inputs)
        correct_pred += correct_prediction(prob, lab_tensor)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy



def train_model(model, ori_model, args, trainset_reader, validset_reader):
    save_path = args.outdir + '/model'
    best_accuracy = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*t_total), num_training_steps=t_total)


    global_step = 0

    model.train()

    for epoch in range(int(args.num_train_epochs)):
        # optimizer.zero_grad()
        for index, data in enumerate(trainset_reader):
            model.train()

            inputs, lab_tensor = data
            prob = model(inputs)
            loss = F.nll_loss(prob, lab_tensor)
            if args.gradient_accumulation_steps > 1:
               loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            global_step += 1
            running_loss += loss.item()

            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                if global_step % (args.gradient_accumulation_steps*10)==0:
                    logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step//args.gradient_accumulation_steps, (running_loss / 10)))
                    running_loss = 0

            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                with torch.no_grad():
                    dev_accuracy = eval_model(model, validset_reader)
                    logger.info('Dev total acc: {0}'.format(dev_accuracy))
                    if dev_accuracy > best_accuracy:
                        best_accuracy = dev_accuracy

                        torch.save({'epoch': epoch,
                                    'model': ori_model.state_dict(),
                                    'best_accuracy': best_accuracy}, save_path + ".best.pt")
                        logger.info("Saved best epoch {0}, best accuracy {1}".format(epoch, best_accuracy))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert", required=True)
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    # parser.add_argument("--bert_hidden_dim", default=1024, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--kernel", type=int, default=21, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=130, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=1000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--postpretrain')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_clip', action='store_true', default=False, help='')

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    set_seed(args)
    logger.info('Start training!')

    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    if args.model_type=='bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
        bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    elif args.model_type=='roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
        bert_model = RobertaForSequenceEncoder.from_pretrained(args.bert_pretrain)
    else:
        assert(False)
    args.bert_hidden_dim = bert_model.hidden_size

    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, label_map, tokenizer, args,
                                 batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, label_map, tokenizer, args,
                                 batch_size=args.valid_batch_size, test=True)

    logger.info('initializing estimator model')

    ori_model = inference_model(bert_model, args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(ori_model)
    else:
        model = ori_model
    model = model.cuda()

    train_model(model, ori_model, args, trainset_reader, validset_reader)
