import torch
import random
import numpy as np

# from vec2seq.dataloader import SentenceLoader
################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "27/07/2020", "1.0"  # create
__date__, __version__ = "29/07/2020", "1.1"  # modify the structure of code
__date__, __version__ = "25/01/2021", "1.2"  # add "added_special_token" in vec2sent

__description__ = "Decoding vectors into sentences"




################################################################################

class decoding():
    def __init__(self,encoder=None,decoder=None,
                    loss_mode="ml",
                    corpora_field=None,max_length=12,device="cpu",
                 init_token=None,eos_token=None,pad_token=None):

        self.device = device

        self.encoder = encoder
        self.decoder = decoder
        self.dec_net_type = self.decoder.net_type


        self.init_token=init_token
        self.eos_token=eos_token
        self.pad_token=pad_token

        self.loss_mode = loss_mode

        self.corpora_field = corpora_field
        self.max_sent_len = max_length #max_original_sent_len + 2

        # self.batchz_size = batchsize

    def sent2index(self,sentence,pad=True):

        # index_list = []
        words = sentence.split(" ")
        if words[0] != self.init_token:
            words = [self.init_token] + words + [self.eos_token]

        if pad:
            assert len(words) <= self.max_sent_len, print("Error: incorrect maximum sentence length setting.")

            words = words + [self.pad_token] * (self.max_sent_len - len(words))


        index_list = [self.corpora_field.stoi[word] for word in words]

        return torch.from_numpy(np.array(index_list)).to(self.device)


    def index2sent(self,index_list, original=False): #original: without the start and end symbols


        sequence = []

        for index in index_list:

            if isinstance(index,int):
                index = index
            else:
                index=index.item()

            if index != self.corpora_field.stoi[self.pad_token]:#PAD_index
                sequence.append(self.corpora_field.itos[index])



        if original: # used to transfer predicted index into sentence
            sent_tmp=[]
            for w in sequence[1:]:
                if w == self.eos_token: #"EOS"
                    break
                else:
                    sent_tmp.append(w)
            sent_tmp = " ".join(sent_tmp)
            return sent_tmp # original sent without SOS and EOS

        else:
            return " ".join(sequence) # including SOS and EOS




    def dec(self,sents_vecs,sents_ref=None, train_mode=True,teacher_forcing_ratio=0.75):# the format of sents_ref must be index
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        if train_mode:
            teacher_forcing_ratio = teacher_forcing_ratio
            # test_sents = None
        else:
            # print(teacher_forcing_ratio)
            teacher_forcing_ratio = 0
            # test_sents=SentenceLoader(test_path)

        # src = [src len, batch size]
        # trg = [trg len, batch size]

        assert sents_vecs.dim() == 3, print("Error: The size of context vectors is wrong!")

        minibatch_size = sents_vecs.size(1)
        vocab_size = self.decoder.output_dim
        if train_mode:
            assert sents_ref.shape[1] == minibatch_size, print("need referenced sents for training")
            sent_len = sents_ref.shape[0]
        else:
            sent_len = self.max_sent_len

        # tensor to store decoder output
        dec_output = torch.zeros(sent_len, minibatch_size, vocab_size).to(self.device)
        batch_output_embeddings = []
        batch_input_embeddings = []

        hidden = sents_vecs
        cell = sents_vecs
        # first input to the decoder is the <sos> tokens
        if train_mode:
            input = sents_ref[0, :]  # trg=[len,batch_size]
        else:
            input = [self.corpora_field.stoi[self.init_token]] * minibatch_size # ["SOS"] * minibatch_size


        for t in range(1, sent_len):

            tokens_batch=[]
            for i in range(minibatch_size):
                if isinstance(input[i], int):
                    tokens_batch.append(self.corpora_field.itos[input[i]])
                else:
                    tokens_batch.append(self.corpora_field.itos[input[i].item()])


            if "bert" in self.encoder.name():
                input_embeddings = self.encoder.sents2vecs(tokens_batch)
            else:
                input_embeddings = [self.encoder.word2vec(token) for token in tokens_batch]
                input_embeddings = torch.cat(input_embeddings, dim=1)



            if self.dec_net_type == "conrnn":
                if self.loss_mode == "ml":
                    batch_input_embeddings.append(input_embeddings)
                    output_embeddings, output, hidden, cell = self.decoder(input_embeddings, hidden, cell,
                                                                      sents_vecs)  # con-lstm
                    batch_output_embeddings.append(output_embeddings)

                elif self.loss_mode == "sl":
                    output, hidden, cell = self.decoder(input_embeddings, hidden, cell, sents_vecs)
                    # outputs_embeddings = None

            elif self.dec_net_type == "basicrnn":
                if self.loss_mode == "ml":
                    batch_input_embeddings.append(input_embeddings)
                    output_embeddings, output, hidden, cell = self.decoder(input_embeddings, hidden, cell)  # con-lstm
                    batch_output_embeddings.append(output_embeddings)

                elif self.loss_mode == "sl":
                    output, hidden, cell = self.decoder(input_embeddings, hidden, cell)
                    # outputs_embeddings = None


            # place predictions in a tensor holding predictions for each token
            dec_output[t] = output  # =[batch_size, trg_vocab_size]
            # batch_output_embeddings.append(output_embeddings)
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # print(top1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = sents_ref[t] if teacher_force else top1


        # eos
        if self.loss_mode == "ml":
            # input = sents_ref[-1, :]
            input = [self.corpora_field.stoi[self.eos_token]] * minibatch_size

            tokens_batch=[]
            for i in range(minibatch_size):
                if isinstance(input[i], int):
                    tokens_batch.append(self.corpora_field.itos[input[i]])
                else:
                    tokens_batch.append(self.corpora_field.itos[input[i].item()])



            if "bert" in self.encoder.name():
                input_embeddings = self.encoder.sents2vecs(tokens_batch)
            else:
                input_embeddings = [self.encoder.word2vec(token) for token in tokens_batch]
                input_embeddings = torch.cat(input_embeddings, dim=1)

            batch_input_embeddings.append(input_embeddings)

            batch_output_embeddings = torch.cat(batch_output_embeddings, dim=0)
            batch_input_embeddings = torch.cat(batch_input_embeddings[1:], dim=0)
        else:
            batch_output_embeddings = None
            batch_input_embeddings = None

        return dec_output, batch_output_embeddings, batch_input_embeddings


    def vec2sent(self, vectors, ref=None, added_special_token=True):

        pred_output_list,output_embeds,input_embeds = self.dec(vectors, train_mode=False, teacher_forcing_ratio=0)
        results = []

        for j in range(pred_output_list.size(1)):
            tmp = pred_output_list[:, j, :]
            index_tmp = tmp.argmax(1)
            sent_tmp = self.index2sent(index_tmp, original=True)
            if ref:

                if not added_special_token:
                    sent_ref = ref[j]
                else:
                    sent_ref = " ".join(ref[j].split(" ")[1:-1])#self.index2sent(ref[j],original=True)#self.index2sent(ref_index_tmp, original=True)

                results.append([sent_ref, sent_tmp])


            else:
                results.append(sent_tmp)


        return results,pred_output_list,output_embeds,input_embeds





