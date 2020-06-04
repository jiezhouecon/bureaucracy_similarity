# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 3/30/20
# Copyright: (c) JieZhou 3/30/20
# ----------------------------------------------

import pickle
import re

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess import prepare_doc2vec, tokenization


with open("leaders.pkl", "rb") as file:
    leaders = pickle.load(file)

with open("code2univ.pkl", "rb") as file:
    code2univ = pickle.load(file)

train = pd.DataFrame(columns=['ccode', 'pcode', 'Title', 'Year', 'Authors', 'Affiliations',
                              'Journal', 'Keywords', 'Abstract', 'Link', 'WFID', 'Leader'])
for code in dict((k, v) for k, v in leaders.items() if k in list(leaders.keys())):
    leader_list = leaders[code]
    univ = code2univ[int(code.split()[0])]

    # Please change this directory accordingly
    df = pd.read_csv('Merged/' + univ + '.csv', engine='python', encoding='utf-8', error_bad_lines=False)
    ccode = re.sub(':', '$', code)

    # Please change this directory accordingly
    with open("Paper List/" + ccode + '.pkl', "rb") as file:
        pubs = pickle.load(file)

    leader_pubs = []
    for leader in leader_list:
        if leader in pubs:
            leader_pubs.extend(pubs[leader])
    faculty_pubs = []
    for faculty in pubs:
        faculty_pubs.extend(pubs[faculty])

    all_pubs = leader_pubs + faculty_pubs
    all_pubs = list(set(all_pubs))
    df['Leader'] = (df.index.isin(leader_pubs))
    train = train.append(df[df.index.isin(all_pubs)], sort=False)

corpus = prepare_doc2vec(train)
tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]

#doc2vec parameters
vector_size = 100
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processes

#pretrained word embeddings
pretrained_emb = "zhwiki_20180420_100d.txt" #None if use without pretrained embeddings


model = Doc2Vec(tagged_data,
                vec_size=vector_size,
                window=window_size,
                min_count=min_count,
                sample=sampling_threshold,
                workers=worker_count, hs=0, dm=dm,
                negative=negative_size,
                dbow_words=1, dm_concat=1,
                pretrained_emb=pretrained_emb, epochs=train_epoch)

model.save("d2v.model")
print("Model Saved")

def cosine(vec1, vec2):
    length1 = np.sqrt(sum([i**2 for i in vec1]))
    length2 = np.sqrt(sum([i**2 for i in vec2]))
    dot_product = sum([vec1[i]*vec2[i] for i in range(len(vec1))])
    return dot_product/(length1 * length2)

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = ["随着共享经济的发展,其不断增加的共享资金规模也成为社会公众关注的重点问题。为了更好地维护消费者资金安全,支持共享经济的持续健康发展,有必要完善共享资金的具体监管政策与措施。",
             "文章基于控股股东的视角,以市场层面的投资者情绪作为IPO择机窗口,实证研究发现,高代理成本控股股东更倾向择机IPO,相比国有控股股东,代理成本特征差异对非国有控股股东IPO择机倾向的影响较弱。这表明代理成本高低对控股股东IPO择机倾向的影响受到制度层面的国有性质因素的制约。2005年IPO询价制度的实施的政策影响研究表明,询价制度的实施弱化了代理成本特征差异对控股股东IPO择机倾向的影响,为监管政策变更影响公司融资行为提供了新兴加转轨市场独特的经验证据。"]
token_test = tokenization(test_data)
v1 = model.infer_vector(token_test[0])
v2 = model.infer_vector(token_test[1])
print(v1)
cos_sim = cosine(v1, v2)
print(cos_sim)