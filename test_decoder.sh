#!/usr/bin/env bash

export data_dir=data/sent/ #/mango/homes/WANG_Liyan/data/vec2seq_toy/
#export train_fn=sent.train.tsv
#export valid_fn=sent.valid.tsv
export test_fp=test.new.tsv #sent.test.tsv



export decoder_path=results/decoder-conrnn-ml_word-vec-sum #decoder_word-vec-sum_toydata



python3 run_decoder.py \
--run test \
--decoder_path $decoder_path \
--data_dir $data_dir \
--test_filename $test_fp \

