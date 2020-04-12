# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 2/17/20
# Copyright: (c) JieZhou 2/17/20
# ----------------------------------------------
import itertools

import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
from gensim.corpora.dictionary import Dictionary
import nltk
from nltk.corpus import stopwords

with open('Chinese_stopwords.txt', encoding='utf8') as f:
    ch_stopwords = f.readlines()
for i in range(len(ch_stopwords)):
    ch_stopwords[i] = re.sub('\\n', '', ch_stopwords[i])

# nltk.download()
eng_stopwords = stopwords.words('english')
stopwords = ch_stopwords + eng_stopwords

stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
print("====> Stopwords and stopflags are loaded.")


def get_common_words(content):
    ret_list = []
    fdist2 = nltk.FreqDist(content)
    most_list = fdist2.most_common(5)
    for x, value in most_list:
        ret_list.append(x)
    return ret_list


def prepare_data(df):
    abstracts = []
    for i in range(df.shape[0]):
        if df.Abstract.isna().iloc[i] == False and type(df.Title.iloc[i]) == str:
            abstracts.append(df.Title.iloc[i] + " " + df.Abstract.iloc[i])
        else:
            abstracts.append(' ')
    return abstracts


def tokenization_text(text):
    # this function is to convert string into words list
    if pd.isnull(text):
        return np.nan
    words = pseg.cut(text)
    line = []

    # drop stop words
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            line.append(word)
    return line


def tokenization(texts):
    result = []
    for text in texts:
        result.append(tokenization_text(text))
    return result


def prepare_lda(df):
    abstracts = prepare_data(df)
    corpus = tokenization(abstracts)  # generate corpus

    dictionary = Dictionary(corpus)  # generate the dictionary based on this corpus

    # Get words with length == 1
    words_len1 = [k for k, v in dictionary.items() if len(v) == 1]
    # words_len1 = []

    # Get words that are too common:
    # fdict = get_common_words(list(itertools.chain(*corpus)))
    # fdict = ['经济', '中国', '发展', '研究', '影响', '经济学']
    fdict = []

    del_ids = [k for k, v in dictionary.items() if v in fdict] + words_len1
    # remove unwanted word ids from the dictionary in place
    dictionary.filter_tokens(bad_ids=del_ids)

    doc_vectors = [dictionary.doc2bow(line) for line in corpus]
    print(fdict)
    return corpus, doc_vectors, dictionary


def prepare_tfidf(df):
    abstracts = prepare_data(df)
    result = []
    for text in abstracts:
        sent = " ".join(tokenization_text(text))
        result.append(sent)
    return result


def prepare_doc2vec(df):

    abstracts = {}
    for i in range(df.shape[0]):
        if df.Abstract.isna().iloc[i] == False and type(df.Title.iloc[i]) == str:
            abstracts[i] = (df.Title.iloc[i] + " " + df.Abstract.iloc[i])

    abstracts = [abstract for abstract in list(abstracts.values())]  # prepare all the text data
    corpus = tokenization(abstracts)  # generate corpus
    return corpus
