# CorefBERT

Source code and dataset for "[Coreferential Reasoning Learning for Language Representation](https://arxiv.org/abs/2004.06870)".

![model](https://github.com/thunlp/CorefBERT/blob/master/model.png)


The code is based on huggaface's [transformers](https://github.com/huggingface/transformers). Thanks to them!

Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1JGDxWqvhWCQ58PqD7dAzye3NW5U2pZOp?usp=sharing)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/a2eed603edd949b0a663).

## Requirement
Install dependencies and [apex](https://github.com/NVIDIA/apex):
```
pip3 install -r requirement.txt
python3 -m spacy download en_core_web_sm
```


## Pre-training

Codes are in the folder 'Pretrain/'.

Download the [Wikipeida database dump](https://dumps.wikimedia.org/enwiki).

Use [WikiExtractor](https://github.com/attardi/wikiextractor) and clean text from the Wikipedia database dump.
```
python3 WikiExtractor.py wikipedia/enwiki-20190820-pages-articles-multistream.xml.bz2 --json --output wikipedia/ --bytes 500M --processes 8
```

Extract nouns from Wikipedia:
```
python3 gen_copy_data_NN.py --train_corpus wikipedia/ --output_dir wikipedia_NN/
```

Figure out the repeated noun, split words to word pieces, masked tokens:
```
python3 gen_copy_data_MRP.py --train_NN_corpus wikipedia_NN/ --output_dir wikipedia_MRP/ --bert_model bert-base-cased --max_seq_length 512 --short_seq_prob 0.1 --max_copy_per_seq 15 
```

Transfer json data to numpy memmap format:
```
python3 gen_copy_data_memmap.py --train_token_corpus wikipedia_MRP/ --output_dir wikipedia_traindata/ --max_seq_length 512 --max_copy_per_seq 15 
```


Train CorefBERT:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 run_pretrain.py --do_train --model_type bert --model_name_or_path bert-base-cased --corpus_path wikipedia_traindata --per_gpu_train_batch_size 8 --learning_rate 5e-5 --num_train_epochs 1  --max_seq_length 512 --save_steps 3000 --logging_steps 5 --warmup_steps 3000 --output_dir ../CorefBERT_base --fp16
```

## QUOREF

Codes are in the folder 'QUOREF/'.


Download the [QUOREF dataset](https://leaderboard.allenai.org/quoref/submissions/get-started) (Questions Requiring Coreferential Reasoning dataset) and put them into the folder 'quoref_data/'.

Baseline code can be found on the [official code](https://github.com/allenai/quoref-leaderboard-example)  for the dataset paper.


We further design two components accounting for coreferential reasoning and multiple answers.

Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad.py --do_train --evaluate_during_training --model_type bert --model_name_or_path ../CorefBERT_base --train_file quoref_data/quoref-train-v0.1.json --predict_file quoref_data/quoref-dev-v0.1.json --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --learning_rate 2e-5 --num_train_epochs 6 --max_seq_length 512 --doc_stride 128 --save_steps 884 --max_n_answers 2 --output_dir QUOREF_CorefBERT_base
```

Evaluate:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_squad.py --do_eval --model_type bert_multi --model_name_or_path ../CorefBERT_base --train_file quoref_data/quoref-train-v0.1.json --predict_file quoref_data/quoref-dev-v0.1.json --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16  --learning_rate 2e-5 --num_train_epochs 6 --max_seq_length 512 --doc_stride 128 --save_steps 884 --max_n_answers 2 --output_dir QUOREF_CorefBERT_base
```

## MRQA

Codes are in the folder 'MRQA/'.


Download the [MRQA dataset](https://github.com/mrqa/MRQA-Shared-Task-2019) and sort out the datasets:
```
sh download_data.sh
```

Randomly split the development set into two halves to generate new validation and test sets:
```
python3 split.py
```

Train (e.g. on SQuAD):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_squad.py --do_train --do_eval --evaluate_during_training  --model_type bert --model_name_or_path ../CorefBERT_base --train_file squad/train.jsonl --predict_file squad/dev.jsonl --doc_stride 128 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 512 --save_steps 1500 --output_dir SQuAD_CorefBERT_base
```

Manualy select the best checkpoint on development set based on the training log and evaluate it on test set:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run_squad.py --do_eval --model_type bert --model_name_or_path ../CorefBERT_base --train_file squad/train.jsonl --predict_file squad/test.jsonl --doc_stride 128 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 512 --save_steps 1500 --output_dir SQuAD_CorefBERT_base
```

## DocRED

Codes are in the folder 'DocRED/'.

We modify the [official code](https://github.com/thunlp/DocRED) to implement BERT-based models.

Download the [DocRED dataset](https://github.com/thunlp/DocRED/tree/master/data) and put them into the folder 'docred_data'.

Preprocess the data:
```
python3 gen_data.py --model_type bert --model_name_or_path bert-base-cased --data_dir docred_data --output_dir prepro_data --max_seq_length 512
```

Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --model_type bert --model_name_or_path ../CorefBERT_base  --train_prefix train --test_prefix dev --evaluate_during_training_epoch 5 --prepro_data_dir prepro_data --max_seq_length 512 --batch_size 32 --learning_rate 4e-5 --num_train_epochs 200 --save_name DocRED_CorefBERT_base
```

Choose the prediction probability threshold based on the training log for filtering the output relational facts and evaluate model on test set with the threshold:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py --model_type bert --model_name_or_path ../CorefBERT_base --test_prefix test --prepro_data_dir prepro_data --max_seq_length 512 --batch_size 32 --save_name DocRED_CorefBERT_base --input_threshold 0.5534
```

## FEVER

Download [FEVER dataset](https://competitions.codalab.org/competitions/18814#learn_the_details-overview) (Fact Extraction and Verification) . We use [code of KGAT](https://github.com/thunlp/KernelGAT) for training and evaluating. Download preprocessed data from [Google Drive](https://drive.google.com/drive/folders/12-0VIoev0PzU4K-IUeWUaVNAlf0ESWZ3?usp=sharing)/[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9ad5c476906041ae9bf7).

Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --model_type bert  --bert_pretrain ../CorefBERT_base  --learning_rate 2e-5  --num_train_epochs 3  --train_path fever_metadata/train.json --valid_path fever_metadata/dev.json  --train_batch_size 32 --valid_batch_size 32  --outdir FEVER_CorefBERT_base
```


Evaluate on dev set:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py  --model_type bert  --outdir output/ --test_path fever_metadata/dev.json --bert_pretrain  ../CorefBERT_base  --batch_size 32 --checkpoint FEVER_CorefBERT_base/model.best.pt --name dev.jsonl
python3 fever_score_test.py --predicted_labels output/dev.jsonl  --predicted_evidence fever_metadata/dev_bert.json --actual fever_metadata/dev.jsonl
```
Evaluate on test set:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py  --model_type bert  --test_path fever_metadata/test.json --bert_pretrain  ../CorefBERT_base --batch_size 32  --checkpoint FEVER_CorefBERT_base/model.best.pt --name test.jsonl 
python3 generate_submit.py --predict output/test.jsonl --original fever_metadata/test.json --order fever_metadata/test_order.json
```


## Coreference resolution

We use [code of WikiCREM](https://github.com/vid-koci/bert-commonsense) for training and evaluating. Coreference resolution datasets can also be found in [WikiCREM](https://github.com/lsvid-koci/bert-commonsense). Thanks to them!



## Cite

If you use the code, please cite this paper:

```
@inproceedings{ye2020corefbert,
  title={Coreferential Reasoning Learning for Language Representation},
  author={Deming Ye, Yankai Lin, Jiaju Du, Zhenghao Liu, Peng Li, Maosong Sun, Zhiyuan Liu},
  booktitle={Proceedings of EMNLP 2020},
  year={2020}
}
```