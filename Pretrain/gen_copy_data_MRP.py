from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from collections import Counter, defaultdict
from random import random, randrange, randint, shuffle, choice
from transformers import BertTokenizer
import numpy as np
import json
import collections
import copy
import spacy
import random
import os


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label", "coref_idx"])

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_NN_corpus', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, default="bert-base-cased")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_copy_per_seq", type=int, default=15, help="The maximum number of tokens using MRP loss")

    args = parser.parse_args()

    max_seq_length = args.max_seq_length

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_words = list(tokenizer.vocab.keys())

    train_NN_corpus = args.train_NN_corpus
    output_dir = args.output_dir
    short_seq_prob = args.short_seq_prob
    masked_lm_prob = args.masked_lm_prob
    max_copy_per_seq = args.max_copy_per_seq

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(os.listdir( train_NN_corpus ), reverse=False)

    for file in files:
        num = int(file[5:-3])
        print(file, num)
        ofile = os.path.join(output_dir, file+".token")
        if os.path.exists(ofile):
            continue

        output_file = open(ofile, "w")
        with open( os.path.join(train_NN_corpus, file) ) as f:
            for line in tqdm(f):
                article = json.loads(line.strip())
                doc = article['info']
                max_num_tokens = max_seq_length - 2
                target_seq_length = max_num_tokens

                i = 0
                while i < len(doc):

                    if random.random() < short_seq_prob:
                        target_seq_length = random.randint(64, max_num_tokens)
                    else:
                        target_seq_length = max_num_tokens

                    tokens = []
                    cand_indexes = []
                    cur_len = 0
                    PRPs = set([])
                    name2pos = defaultdict(list)

                    for j in range(i, len(doc)):
                        token, tag = doc[j]
                        is_Noun = (tag == 'NNS' or tag == 'NN' or tag == 'NNPS' or tag == 'NNP') and (len(token)>1)
                        subwords = tokenizer.tokenize(token)
                        tokens.extend(subwords)

                        if cur_len + len(subwords)  > target_seq_length:  
                            break

                        cand_indexes.append((cur_len+1, cur_len + 1 + len(subwords)))  
                        assert(cand_indexes[-1][1] <= max_seq_length-1)
                        if tag == 'PRP':
                            for x in range(cand_indexes[-1][0], cand_indexes[-1][1]):
                                PRPs.add( x )
                        if is_Noun:
                            name2pos[token.lower()].append ( cand_indexes[-1] )
                        
                        cur_len += len(subwords)

                        if cur_len == target_seq_length:  
                            break


                    next_i = j + 1
                    if len(tokens) < 62:
                        break


                    name2pos = [(k, v) for k,v in name2pos.items() if len(v)>1]
                    tokens = ['[CLS]'] + tokens[:target_seq_length] + ['[SEP]']
                    output_tokens = list(tokens)
                    num_to_predict = max(1, int(round(len(tokens) * masked_lm_prob)))
                    num_to_copy = min(max_copy_per_seq, max(1, round(num_to_predict*0.2)))

                    random.shuffle(name2pos)
                    random.shuffle(cand_indexes)

                    masked_lms = []
                    covered_indexes = set()

                    for k, vlist in name2pos:
                        random.shuffle(vlist)
                        v = vlist[-1]
                        if len(masked_lms) + v[1]-v[0] > num_to_copy:
                            continue

                        for idx, index in enumerate(range(v[0], v[1])):
                            covered_indexes.add(index)

                            masked_token = None
                            # 80% of the time, replace with [MASK]
                            if random.random() < 0.8:
                                masked_token = "[MASK]"
                            else:
                                # 10% of the time, keep original
                                if random.random() < 0.5:
                                    masked_token = tokens[index]
                                # 10% of the time, replace with random word
                                else:
                                    masked_token = vocab_words[random.randint(1, len(vocab_words) - 1)]

                            output_tokens[index] = masked_token

                            if index==v[0]: 
                                coref_idx = [x for x, y in vlist[:-1]]
                            elif index==v[1]-1:
                                coref_idx = [y-1 for x, y in vlist[:-1]]
                            else:
                                coref_idx = []
                            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index], coref_idx=coref_idx))
        
                    for index_set in cand_indexes:
                        if len(masked_lms) + index_set[1]-index_set[0] > num_to_predict:
                            continue

                        if index_set[0] in covered_indexes:
                            continue
                        if index_set[1]-1 in covered_indexes:
                            continue

                        for index in range(index_set[0], index_set[1]):
                            masked_token = None
                            # 80% of the time, replace with [MASK]
                            if random.random() < 0.8:
                                masked_token = "[MASK]"
                            else:
                                # 10% of the time, keep original
                                if random.random() < 0.5:
                                    masked_token = tokens[index]
                                # 10% of the time, replace with random word
                                else:
                                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

                            output_tokens[index] = masked_token

                            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index], coref_idx=[]))

                    assert len(masked_lms) <= num_to_predict
                    masked_lms = sorted(masked_lms, key=lambda x: x.index)

                    masked_lm_positions = []
                    masked_lm_labels = []
                    coref_idxs = []
                    for p in masked_lms:
                        masked_lm_positions.append(p.index)
                        masked_lm_labels.append(p.label)
                        coref_idxs.append(p.coref_idx)


                    input_ids = tokenizer.convert_tokens_to_ids(output_tokens)
                    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
                    
                    item = {
                        'input_ids': input_ids,
                        'masked_lm_positions': masked_lm_positions,
                        'masked_lm_ids': masked_lm_ids,
                        'coref_idxs': coref_idxs
                    }
                    output_file.write(json.dumps(item)+'\n')

                    i = next_i



if __name__ == '__main__':
    main()
