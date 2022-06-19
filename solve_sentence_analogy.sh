#!/usr/bin/env bash


export decoder_path=results/decoder-conrnn-ml_word-vec-sum #de/coder_word-vec-sum_toydata
export solver_path=results/solver_ap_word-vec-sum #solver_ap_word-vec-sum_toydata

export encoder_name=word-vec-sum
export encoder_path=/mango/homes/WANG_Liyan/model/cc.en.300.init.bin

export analogy_test_path=/mango/homes/WANG_Liyan/code/vec2seq/data/analogy/tatoeba.en.lowercase.test.nlg # /mango/homes/WANG_Liyan/data/vec2seq_toy/nlg.test.tsv
export test_filepath=data/nlg_${encoder_name}.test.npz

if [ ! -f $test_filepath ]; then
  {
    python3 preprocessing.py --encoder_name $encoder_name --encoder_path $encoder_path \
      --input_path $analogy_test_path --output_path $test_filepath

  }; fi





export save_filepath=results/ap_conrnn_ml_word_vec_sum.tsv

python3 run_AnalogySolver.py --decoder_path $decoder_path \
--solver_path $solver_path \
--test_filepath $test_filepath \
--save_filepath $save_filepath
