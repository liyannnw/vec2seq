#!/usr/bin/env bash

#@dataset
export data_dir=data/sent/ #/mango/homes/WANG_Liyan/data/vec2seq_toy/ #/homes/WANG_Liyan/code/vec2seq/data/sentence/
export train_fn=train.new.tsv
export valid_fn=valid.new.tsv #nlg.valid.tsv #
export test_fn=test.new.tsv

#@encoding method
#export encoder_name=bert-base-nli-mean-tokens
#export encoder_path=None
export encoder_name=word-vec-sum
export encoder_path=/mango/homes/WANG_Liyan/model/cc.en.300.init.bin

#@decoder
export decoder_loss=ml #sl #
export decoder_net=conrnn #basicrnn #

#export epoch=1

if [ ! -d results ]; then mkdir ./results ;fi
export model_save_path=results/decoder-${decoder_net}-${decoder_loss}_${encoder_name}



python3 run_decoder.py \
--data_dir $data_dir \
--train_filename $train_fn \
--valid_filename $valid_fn \
--test_filename $test_fn \
--encoder_name $encoder_name \
--model_save_path $model_save_path \
--decoder_net $decoder_net \
--decoder_loss $decoder_loss \
--encoder_path $encoder_path \
#--epochs $epoch \
