# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 4/24/20
# Copyright: (c) JieZhou 4/24/20
# ----------------------------------------------
import re
import os
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
import pickle
from gensim.corpora import Dictionary

dtm_path = "/Users/JieZhou/Desktop/similarity/dtm/dtm/main"

with open("data/leaders.pkl", "rb") as file:
    leaders = pickle.load(file)

univ_with_leader_info = dict((k, v) for k, v in leaders.items() if k in list(leaders.keys()))

code = list(univ_with_leader_info.keys())[0]
dept_dict = Dictionary.load("doc-vec/{}.dict".format(code))

with open('doc-vec/{}-seq.dat'.format(code), 'r') as file:
    time_seq = file.readlines()
# Change time_seq from string to list
time_seq = [int(re.findall(r'\d+', str)[0]) for str in time_seq]

with open("doc-vec/{}.pkl".format(code), "rb") as file:
    doc_vec = pickle.load(file)

corpus = []
for doc in doc_vec.values():
    if not doc:
        print(doc)
    else:
        corpus.append([(v, k) for k, v in doc.items()])
print(corpus)
model = DtmModel(dtm_path, corpus, time_seq, num_topics=100,
                     id2word=dept_dict, initialize_lda=True, model='fixed')
save_dir = 'result/%s' % code
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save(save_dir + "/dim")