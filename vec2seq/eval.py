#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
# from vec2seq.EditDistance import editdistance
import sacrebleu
# from vec2seq.similarity import distance
import distance
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


            dist_w = distance.levenshtein(line_sents[1].split(), line_sents[0].split())
            dist_c = distance.levenshtein(line_sents[1], line_sents[0])
            dist.append([dist_w,dist_c])
            if line_sents[1] == line_sents[0]:
                acc +=1
    Nw = [value[0] for value in dist]
    Nc = [value[1] for value in dist]

    ref= list(np.array(sentlist)[:,0])
    pred = list(np.array(sentlist)[:,1])

    
    sacre_bleu = sacrebleu.corpus_bleu(pred, [ref])


    scores = {'BLEU': np.round(sacre_bleu.score,1),
    "Accuracy": np.round(100*acc/num,1),
    "Distance in words": np.round(np.mean(Nw),1),
    "Distance in chars":np.round(np.mean(Nc),1)}

    return scores

