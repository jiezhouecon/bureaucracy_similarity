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
from numpy import dot
from numpy.linalg import norm

from bert_serving.client import BertClient
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import HdpModel, TfidfModel
from gensim.matutils import sparse2full, jensen_shannon, cossim, hellinger
from gensim.corpora import Dictionary
from gensim.models.wrappers import DtmModel


class calculator(object):
    """Calculate similarity score

        Parameters
        ----------

        Returns
        -------
        float
           similarity measure for `index`.
    """
    local_ip = socket.gethostbyname(socket.gethostname())

    # Define stopwords and stop_flag:
    with open('Chinese_stopwords.txt', encoding='utf8') as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = re.sub('\\n', '', stopwords[i])

    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    print("====> Stopwords and stopflags are loaded.")

    def __init__(self, df1, df2, abs=True, token=True):
        self.df1 = df1
        self.df2 = df2

        self.abstracts1 = {}
        self.abstracts2 = {}
        if abs == True:
            for i in range(len(self.df1)):
                if df1.Abstract.isna().iloc[i] == False and type(df1.Title.iloc[i]) == str:
                    self.abstracts1[i] = (df1.Title.iloc[i] + " " + df1.Abstract.iloc[i])
                else:
                    self.abstracts1[i] = (df1.Title.iloc[i])
            for i in range(len(self.df2)):
                if df2.Abstract.isna().iloc[i] == False and type(df2.Title.iloc[i]) == str:
                    self.abstracts2[i] = (df2.Title.iloc[i] + " " + df2.Abstract.iloc[i])
                else:
                    self.abstracts2[i] = (df2.Title.iloc[i])

        if token == True:
            # Tokenize abstracts
            self.corpus1 = self.tokenize(self.abstracts1)
            self.corpus2 = self.tokenize(self.abstracts2)
            print("====> Corpus is prepared.")
        else:
            self.corpus1 = []
            self.corpus2 = []

    def tokenize(self, abstracts):
        result = []
        for text in list(abstracts.values()):
            words = pseg.cut(text)
            line = []
            for word, flag in words:
                if flag not in self.stop_flag and word not in self.stopwords:
                    line.append(word)
            result.append(line)
        return result

    def prep_bert(self, model_dir, wait_time):
        """Load BERT model"""
        # subprocess.call(["pip", "install", "-U", "bert-serving-server", "bert-serving-client"])
        # subprocess.call(["bert-serving-terminate", "-ip", self.local_ip, "-port", "8019"])
        bert_start = subprocess.Popen(["bert-serving-start",
                                       "-model_dir", model_dir,
                                       "-num_worker=30",
                                       "-port", "8019", "-cpu"],
                                      stdout=subprocess.PIPE)
        time.sleep(wait_time)
        print("====> BERT is loaded.")

    def bert_cos(self, indices):
        """Return the similarity score based on BERT model."""
        left_term = list(self.abstracts1.values())[indices[0]]
        right_term = list(self.abstracts2.values())[indices[1]]
        with BertClient(ip=self.local_ip, port=8019, check_length=False) as bc:
            a = bc.encode([left_term])
            b = bc.encode([right_term])
            cos_sim = cosine_similarity(a, b)[0, 0]
        return cos_sim

    def prep_doc2vec(self):
        # Load Doc2Vec model
        self.d2v_model = Doc2Vec.load("d2v.model")
        print("====> Doc2Vec is loaded.")

    def doc2vec(self, indices):
        """Return the similarity score based on doc2vec model."""
        sim = self.d2v_model.docvecs.similarity_unseen_docs(self.d2v_model,
                                                            self.corpus1[indices[0]],
                                                            self.corpus2[indices[1]])
        return sim

    def prep_lda(self, dic_dir, model_dir):
        """Load LDA model"""
        # Load LDA model
        self.lda_model = LdaMulticore.load(model_dir)
        print("====> LDA is loaded.")

        self.lda_dict = Dictionary.load(dic_dir)
        print("====> Dictionary is prepared.")

    def lda(self, indices):
        """Return the similarity score based on LDA model."""
        doc1 = self.lda_dict.doc2bow(self.corpus1[indices[0]])
        doc2 = self.lda_dict.doc2bow(self.corpus2[indices[1]])
        vec1 = self.lda_model.get_document_topics(doc1, minimum_probability=0)
        vec2 = self.lda_model.get_document_topics(doc2, minimum_probability=0)
        print(vec1, vec2)
        sim = jensen_shannon(vec1, vec2)
        return sim

    def prep_hdp(self, dic_dir, model_dir):
        """Load LDA model"""
        # Load LDA model
        self.hdp_model = HdpModel.load(model_dir)
        print("====> LDA is loaded.")

        self.hdp_dict = Dictionary.load(dic_dir)
        print("====> Dictionary is prepared.")

    def hdp(self, indices):
        """Return the similarity score based on LDA model."""
        doc1 = self.hdp_dict.doc2bow(self.corpus1[indices[0]])
        doc2 = self.hdp_dict.doc2bow(self.corpus2[indices[1]])
        vec1 = sparse2full(self.hdp_model[doc1], length=150)
        vec2 = sparse2full(self.hdp_model[doc2], length=150)

        sim = jensen_shannon(vec1, vec2)
        return sim

    def prep_tfidf(self, dic_dir, model_dir):
        """Load TF-IDF model"""
        self.tfidf_model = TfidfModel.load(model_dir)
        print("====> TF-IDF is loaded.")

        self.tfidf_dict = Dictionary.load(dic_dir)
        print("====> Dictionary is prepared.")

    def tfidf(self, indices):
        doc1 = self.tfidf_dict.doc2bow(self.corpus1[indices[0]])
        doc2 = self.tfidf_dict.doc2bow(self.corpus2[indices[1]])
        v1 = self.tfidf_model[doc1]
        v2 = self.tfidf_model[doc2]
        sim = cossim(v1, v2)
        return sim

    def prep_dtm(self, dic_dir = 'dim/model', model_dir = 'dim/corpus.csv'):
        """Load LDA model"""
        self.dtm = DtmModel.load(dic_dir)
        print("====> DTM is loaded.")

        self.traindata = pd.read_csv(model_dir)
        self.traindata['within_year'] = self.traindata.groupby(['Year']).cumcount()
        print("====> Dictionary is prepared.")

        self.year_list = sorted(list(set(self.traindata['Year'].to_list())))

    def dtm_cos(self, indices, univs):
        ccode1 = re.sub(':', '$', univs[0])
        print(ccode1)
        paper1 = self.traindata[(self.traindata['idx'] == indices[0]) &
                                (self.traindata['univ_code'] == ccode1)]
        print(paper1)

        doc1 = paper1.index.values.astype(int)[0]

        ccode2 = re.sub(':', '$', univs[1])
        paper2 = self.traindata[(self.traindata['idx'] == indices[1]) &
                                (self.traindata['univ_code'] == ccode2)]
        print(paper2)

        doc2 = paper2.index.values.astype(int)[0]

        v1 = self.dtm.gamma_[doc1, :]
        v2 = self.dtm.gamma_[doc2, :]

        sim = dot(v1, v2)/(norm(v1)*norm(v2))
        return sim
