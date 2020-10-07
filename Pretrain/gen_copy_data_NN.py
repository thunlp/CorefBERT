from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from collections import Counter, defaultdict
# import random
# random.seed(6)

from random import random, randrange, randint, shuffle, choice
from transformers import BertTokenizer
import numpy as np
import json
import collections
from nltk.tokenize import sent_tokenize
import copy
import spacy
import random
import os
from bs4 import BeautifulSoup as bs

nlp = spacy.load('en_core_web_sm', disable=[ "parser", "ner"])



MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label", "coref_idx"])


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)

    args = parser.parse_args()
    
    train_corpus = os.path.join(args.train_corpus, 'AA')
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(os.listdir( train_corpus ), reverse=False)

    for file in files:
        num = int(file[5:])
        print(file, num)
        ofile = os.path.join(output_dir, file+".NN")
        if os.path.exists(ofile):
            continue

        output_file = open(ofile, "w")
        with open( os.path.join(train_corpus, file) ) as f:
            for line in tqdm(f):
                article = json.loads(line.strip())
                title = article['title']
                if title.find('(disambiguation)') != -1:
                    continue

                soup = bs(article['text'], "html.parser")
                text = soup.text.strip()
                text = text.strip()

                doc = nlp(text)
                item = [(token.text, token.tag_) for token in doc]
                item = {'title': title,  'info':item}
                output_file.write(json.dumps(item)+'\n')

              

if __name__ == '__main__':
    main()
