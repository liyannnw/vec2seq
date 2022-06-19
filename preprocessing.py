#!/usr/bin/env python3
from vec2seq.sentence2vector import encoding
import numpy as np
from argparse import ArgumentParser
import os


################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "16/01/2021", "1.0"


__usage__='''
'''

################################################################################
def read_argv():
    parser=ArgumentParser(usage=__usage__)


    parser.add_argument("--encoder_name", dest='encoder_name', action="store",default=None)
    parser.add_argument("--encoder_path", dest='encoder_path', action="store",default=None)


    parser.add_argument("--input_path",action="store",dest="input_path",default=None)
    parser.add_argument("--output_path",action="store",dest="output_path",default=None)


    return parser.parse_args()



################################################################################
def DataProcessing(encoder=None,input_path=None,output_path=None):
    '''
    To encode sentences as vectors,
    and save analogy indices, sentences, vectors into .npz file
    '''

    s2i_dict = dict()
    nlg_indices = list()
    data_size = 0
    embeddings =list()
    sents=list()

    with open(input_path, "r+") as file:
        for line in file:
            data_size += 1
            line = line.strip("\n").split("\t")

            for term in line:
                term_SE = " ".join(["SOS"] + term.split(" ") + ["EOS"])
                vector = encoder.sent2vec(term_SE).squeeze().numpy()
                if term not in s2i_dict:
                    s2i_dict[term] = len(s2i_dict)
                    sents.append(term)
                    embeddings.append(vector)


            indexes = [s2i_dict[sent] for sent in line]
            nlg_indices.append(indexes)

    assert data_size == len(nlg_indices)
    # print("{} analogies/{} sentences".format(len(nlg_indices),len(sents)))

    np.savez(output_path,nlg_indices=nlg_indices,sentences=sents,vectors=embeddings)


################################################################################
if __name__ == '__main__':

    opt = read_argv()


    print(">> Loading the embedding model...")

    encoder = encoding(model_name=opt.encoder_name,model_path=opt.encoder_path, device='cpu')




    print(">> Processing textual analogies and save to a .npz file...\n")
    # output_path = os.path.splitext(opt.input_path)[0]

    DataProcessing(encoder=encoder,input_path=opt.input_path,output_path=opt.output_path)

    print(">> Done.")


