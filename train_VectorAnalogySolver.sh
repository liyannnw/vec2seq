#!/usr/bin/env bash


#configurations
export data_dir=/mango/homes/WANG_Liyan/code/vec2seq/data/analogy #/mango/homes/WANG_Liyan/data/vec2seq_toy

export analogy_train_path=$data_dir/tatoeba.en.lowercase.train.nlg #nlg.train.tsv
export analogy_valid_path=$data_dir/tatoeba.en.lowercase.valid.nlg #nlg.valid.tsv


export encoder_name=word-vec-sum
export encoder_path=/mango/homes/WANG_Liyan/model/cc.en.300.init.bin


export processed_data_dir=data
if [ ! -d $processed_data_dir ]; then mkdir ./$processed_data_dir ;fi


export train_filepath=$processed_data_dir/nlg_${encoder_name}.train.npz
export valid_filepath=$processed_data_dir/nlg_${encoder_name}.valid.npz


export compose_mode=ap

#export epochs=1

if [ ! -d results ]; then mkdir ./results ;fi
export model_save_path=results/solver_${compose_mode}_${encoder_name}





# preprocessing: encode sentences in analogies to vectors and save in a .npz file

if [ ! -f $train_filepath ]; then
  {
    python3 preprocessing.py --encoder_name $encoder_name --encoder_path $encoder_path \
      --input_path $analogy_train_path --output_path $train_filepath

  }; fi


if [ ! -f $valid_filepath ]; then
  {
    python3 preprocessing.py --encoder_name $encoder_name --encoder_path $encoder_path \
      --input_path $analogy_valid_path --output_path $valid_filepath

  }; fi




# training a vector analogy solver
if [ ! -f model_save_path ]; then
  {
    python3 run_VectorAnalogySolver.py --train_filepath ${train_filepath} \
    --valid_filepath ${valid_filepath} \
    --compose_mode ${compose_mode} \
    --model_save_path $model_save_path \
#     --epochs $epochs \


  }; fi



