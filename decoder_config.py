#!/usr/bin/env python3
# -*- coding: utf-8 -*-

run = "train"


data_dir = "/mango/homes/WANG_Liyan/data/vec2seq_toy/"
train_filename = "sent.train.tsv"
valid_filename = "sent.valid.tsv"
test_filename = "sent.test.tsv"


BATCH_SIZE = 128
N_EPOCHS = 1000
CLIP = 1
PATIENCE = 50
N_LAYERS = 1
DEC_DROPOUT = 0.4

teacher_forcing_ratio = 0.75

decoder_loss = "sl" # {"sl", "ml}
decoder_net = "basicrnn" # {"conrnn", "basicrnn"}


start_of_speech = "SOS"
end_of_speech = "EOS"
padding_token = "PAD"
max_original_sent_len = 10
fixed_length = max_original_sent_len + 2


#
embed_method = "sbert" # {"fasttext","sbert"}
embed_mode = None # {"sum", "avg", None}
embed_model_name_or_path = "bert-base-nli-mean-tokens"

# embed_method = "fasttext" # {"fasttext","sbert"}
# embed_mode = "sum" # {"sum", "avg", None}
# embed_model_name_or_path = "/mango/homes/WANG_Liyan/model/cc.en.300.init.bin"


result_dir = "results/decoder-"
save_path = result_dir + "_".join([embed_method, decoder_net, decoder_loss])+".pt"


pretrained_decoder_path=None


