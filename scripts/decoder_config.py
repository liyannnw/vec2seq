
# run = "train"
run = "test"


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
# max_original_sent_len = 10
max_length = 12


# #
# embed_method = "sbert" # {"fasttext","sbert"}
# embed_mode = None # {"sum", "avg", None}
# encoder_name = "bert-base-nli-mean-tokens"
# encoder_path = None

# # embed_method = "fasttext" # {"fasttext","sbert"}
# # embed_mode = "sum" # {"sum", "avg", None}
encoder_name = "word-vec-sum"
encoder_path = "/mango/homes/WANG_Liyan/model/cc.en.300.init.bin"


result_dir = "results/vec2seq-"
model_save_path = result_dir + "_".join([encoder_name, decoder_net, decoder_loss])+".pt"


pretrained_decoder_path=model_save_path
sentence_save_path= "results/sents-"+"_".join([encoder_name, decoder_net, decoder_loss])+".tsv"

