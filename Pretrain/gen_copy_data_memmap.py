from argparse import ArgumentParser
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm, trange
import os
import pickle
def main():
    parser = ArgumentParser()

    parser.add_argument('--train_token_corpus', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_copy_per_seq", type=int, default=15,  help="The maximum number of tokens using MRP loss") 
    parser.add_argument("--max_coref_np", type=int, default=8, help="the maximum numbers of coreference instance stored in numpy memmap, the others are stored in pickle file") 

    args = parser.parse_args()

    train_token_corpus = args.train_token_corpus
    output_dir = args.output_dir
    max_seq_length = args.max_seq_length
    max_copy_per_seq = args.max_copy_per_seq
    max_coref_np = args.max_coref_np

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(os.listdir( train_token_corpus ), reverse=False)

    num_samples = 0
    for file in files:
        with open( os.path.join(train_token_corpus, file) ) as f:
            num_samples += len(f.readlines())



    input_ids = np.memmap(filename= os.path.join(output_dir, 'input_ids.memmap'), shape=(num_samples, max_seq_length), mode='w+', dtype=np.int32)
    span_start = np.memmap(filename= os.path.join(output_dir, 'span_start.memmap'), shape=(num_samples, max_copy_per_seq), mode='w+', dtype=np.int32)
    lm_labels = np.memmap(filename= os.path.join(output_dir, 'lm_labels.memmap'), shape=(num_samples,  max_seq_length), mode='w+', dtype=np.int32)
    labels = np.memmap(filename= os.path.join(output_dir, 'labels.memmap'), shape=(num_samples, max_copy_per_seq, max_coref_np), mode='w+', dtype=np.int32)

    span_start[:] = 0
    lm_labels[:] = -1
    labels[:] = 0

    i = 0
    all_label_masks = []

    def add_file(file_name):
        nonlocal i 
        f = open(file_name)
        for line in tqdm(f, desc="Loading Dataset "+str(file_name), unit=" lines"):
            line = json.loads(line.strip())
            input_id = line['input_ids']
            masked_lm_positions = line['masked_lm_positions']
            masked_lm_ids = line['masked_lm_ids']

            coref_idxs = line['coref_idxs']

            while len(input_id) < max_seq_length:
                input_id.append(0)
            
            input_ids[i] = input_id
            lm_labels[i][masked_lm_positions] = masked_lm_ids

            assert(len(masked_lm_positions)==len(coref_idxs))
            k = 0
            ins_label_masks = []
            for p, coref_idx in zip(masked_lm_positions, coref_idxs):
                if len(coref_idx)>0:
                    flag = False
                    label_masks = []
                    for idx in coref_idx:
                        if idx < max_seq_length:
                            flag = True
                            label_masks.append(idx)

                    if flag:
                        span_start[i][k] = p
                        for z in label_masks[max_coref_np: ]:
                            ins_label_masks.append( (k, z) )
                        for z_idx, z in enumerate(label_masks[: max_coref_np]):
                            labels[i][k][z_idx] = z

                        k += 1 


            assert(len(masked_lm_positions)==len(masked_lm_ids))
            all_label_masks.append(ins_label_masks)

            i += 1    
    

    for file in files:
        file_name = os.path.join(train_token_corpus, file) 
        add_file(file_name)

    pickle.dump(all_label_masks, open(os.path.join(output_dir, "other_label_masks.pkl"), "wb"))

    assert (i == num_samples)
    
    config = {"num_samples": num_samples, 
                "max_seq_length": max_seq_length, 
                "max_copy_per_seq": max_copy_per_seq, 
                "max_coref_np": max_coref_np}
    json.dump(config, open(os.path.join(output_dir, "data_config.json"), "w"))


if __name__ == '__main__':
    main()
