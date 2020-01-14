# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: similarity
# Purpose: Class for calculating similarity scores
#
# Author: JieZhou
#
# Created: 2020-01-12
# Copyright: (c) JieZhou 2020-01-12
# ----------------------------------------------
import re
import socket
import subprocess
import time

import jieba.posseg as pseg
import pandas as pd
from bert_serving.client import BertClient
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

local_ip = socket.gethostbyname(socket.gethostname())


class similarity(object):
    ''' Calculate similarity score
        input: should be a tupe of index
    '''

    # Define stopwords and stop_flag:
    with open('Chinese_stopwords.txt', encoding='utf8') as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = re.sub('\\n', '', stopwords[i])

    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    print("====> Stopwords and stopflags are loaded.")

    def __init__(self, df):
        self.df = df

        self.abstracts = {}
        for i in range(len(self.df)):
            if self.df.Abstract.isna().iloc[i] == False and type(self.df.Title.iloc[i]) == str:
                self.abstracts[i] = (self.df.Title.iloc[i] + "。" + self.df.Abstract.iloc[i])

        self.corpus = []

    def tokenize(self):
        result = []
        for text in list(self.abstracts.values()):
            words = pseg.cut(text)
            line = []
            for word, flag in words:
                if flag not in self.stop_flag and word not in self.stopwords:
                    line.append(word)
            result.append(line)
        return result

    def prep_bert(self):
        """Load BERT model"""
        subprocess.check_output(["pip", "install", "-U", "bert-serving-server", "bert-serving-client"])
        bert_start = subprocess.Popen(["bert-serving-start",
                                       "-model_dir", "model/chinese_L-12_H-768_A-12",
                                       "-num_worker=1",
                                       "-port", "8019"],
                                      stdout=subprocess.PIPE)
        time.sleep(30)
        # "bert-serving-terminate -ip 10.189.90.86 -port 8019"
        print("====> BERT is loaded.")

    def bert_cos(self, indices):
        """Return the similarity score based on BERT model."""
        print("====> Start to calculate BERT similarity.")
        left_term = list(self.abstracts.values())[indices[0]]
        right_term = list(self.abstracts.values())[indices[1]]
        with BertClient(ip=local_ip, port=8019, check_length=False) as bc:
            a = bc.encode([left_term])
            b = bc.encode([right_term])
            cos_sim = cosine_similarity(a, b)
        return cos_sim

    def prep_doc2vec(self):
        # Tokenize abstracts
        self.corpus = self.tokenize()
        print("====> Corpus is prepared.")

        # Load Doc2Vec model
        self.d2v_model = Doc2Vec.load("d2v.model")
        print("====> Doc2Vec is loaded.")

    def doc2vec(self, indices):
        """Return the similarity score based on doc2vec model."""
        # if len(self.corpus) == 2:
        sim = self.d2v_model.docvecs.similarity_unseen_docs(self.d2v_model,
                                                            self.corpus[indices[0]],
                                                            self.corpus[indices[1]])
        return sim
        # else:
        #    print("You should tokenize texts using similarity.tokenize() first before calculate the doc2vec measure.")

    # def lsi(self,):


df = pd.read_csv('南开大学.csv')
df = df[df['Affiliations'].str.contains("经济")].reset_index(drop=True)
df = df.head(20)

# Initialize the similarity calculator
sim_calculator = similarity(df)
# Prepare the corresponding model for the measure
sim_calculator.prep_bert()
# Calculate similarity
print(sim_calculator.bert_cos((9, 8)))
