#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# from VectorDecoder.EditDistance import editdistance
import sacrebleu
from VectorDecoder.similarity import distance
################################################################################
__author__ = "WANG Liyan <wangliyan0905@toki.waseda.jp>"
__date__, __version__ = "26/03/2020", "1.0"  # create

__description__ = """Evaluation (BLEU, accuracy and edit distances in words and characters).

"""
################################################################################


def eval(sentlist):

    dist=[]
    acc=0
    num,l = np.shape(sentlist)
    if l == 2:
        for line_sents in sentlist:
            line_ref=[[line_sents[0]]]
            line_pred = [line_sents[1]]
            line_bleu = sacrebleu.corpus_bleu(line_pred, line_ref)


            dist_w = distance(line_sents[1], line_sents[0],method='Hunt and Szymanski', word_unit=True)
            dist_c = distance(line_sents[1], line_sents[0],method='Hunt and Szymanski', word_unit=False)
            dist.append([dist_w,dist_c])
            if line_sents[1] == line_sents[0]:
                acc +=1
    Nw = [value[0] for value in dist]
    Nc = [value[1] for value in dist]

    ref= list(np.array(sentlist)[:,0])
    pred = list(np.array(sentlist)[:,1])

    ref = [ref]
    sacre_bleu = sacrebleu.corpus_bleu(pred, ref)

    print("bleu: {}".format(sacre_bleu.score))
    print("acc: {:.3f}".format(100*acc/num))
    print("dist (word): {:.3f}+-{:.3f}".format(np.mean(Nw),np.std(Nw)))
    print("dist (char): {:.3f}+-{:.3f}".format(np.mean(Nc), np.std(Nc)))
