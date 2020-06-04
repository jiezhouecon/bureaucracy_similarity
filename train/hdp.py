# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 2/19/20
# Copyright: (c) JieZhou 2/19/20
# ----------------------------------------------

import re
import pickle
import pandas as pd
import numpy as np
from gensim.models import HdpModel
from preprocess import prepare_lda
from gensim.models.ldamulticore import LdaModel
import pyLDAvis.gensim

def topic_prob_extractor(model):
    shown_topics = model.show_topics(num_topics=200, formatted=False)
    topics_nos = [x[0] for x in shown_topics]
    weights = [sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos]
    return pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights})

with open("leaders.pkl", "rb") as file:
    leaders = pickle.load(file)

with open("code2univ.pkl", "rb") as file:
    code2univ = pickle.load(file)

train = pd.DataFrame(columns=['ccode', 'pcode', 'Title', 'Year', 'Authors', 'Affiliations',
                              'Journal', 'Keywords', 'Abstract', 'Link', 'WFID'])
for code in dict((k, v) for k, v in leaders.items() if k in list(leaders.keys())):
    leader_list = leaders[code]
    univ = code2univ[int(code.split()[0])]

    # Please change this directory accordingly
    df = pd.read_csv('Merged/' + univ + '.csv', engine='python', encoding='utf-8', error_bad_lines=False)
    ccode = re.sub(':', '$', code)
    print(ccode, univ)

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

    train = train.append(df[df.index.isin(all_pubs)], sort=False)

method = 'est'
if method == 'update':
    _, _, dictionary = prepare_lda(train)
    dictionary.save('ldaModel/dict_hdp.dict')

    train_year = train[2000 >= train['Year']]
    _, doc_vectors, _ = prepare_lda(train_year, dict=dictionary)
    hdp_model = HdpModel(doc_vectors, dictionary)
    print(hdp_model)
    #hdp_model.save('hdpModel/hdp_y%d' % 2000)

    for year in range(2001, 2020):
        train_year = train[year == train['Year']]
        # train_year = train_year.sample(n=size)

        _, doc_vectors, _ = prepare_lda(train_year, dict=dictionary)
        # dictionary.save('ldaModel/dict_hdp_y%d.dict' % year)
        print(doc_vectors)
        hdp_model = hdp_model.update(doc_vectors)
        print("The model is:", hdp_model)
        #hdp_model.save('hdpModel/hdp_y%d' % year)
else:
    _, _, dictionary = prepare_lda(train)
    dictionary.save('hdpModel/dict_hdp.dict')

    for year in range(2000, 2020):
        train_year = train[year + 3 >= train['Year']]
        # train_year = train_year.sample(n=size)
        print(len(train_year))
        _, doc_vectors, _ = prepare_lda(train_year, dict=dictionary)
        # dictionary.save('hdpModel/dict_hdp_y%d.dict' % year)

        hdp_model = HdpModel(doc_vectors, dictionary)
        hdp_model.save('hdpModel/hdp_y%d' % year)

'''
texts, doc_vectors, dictionary = prepare_lda(train)
dictionary.save('ldaModel/dict_hdp.dict')
hdp = HdpModel(doc_vectors, dictionary)
hdp.save('ldaModel/lda_hdp')


alpha, beta = hdp.hdp_to_lda()
print(alpha, beta)
lda = LdaModel(id2word=hdp.id2word, num_topics=len(alpha), alpha=alpha, eta=hdp.m_eta)
lda.Elogbeta = np.array(beta, dtype=np.float32)

visualisation = pyLDAvis.gensim.prepare(lda, doc_vectors, dictionary=dictionary)
pyLDAvis.save_html(visualisation, 'visLDA/LDA_Visualization_19_nolen1_hdp.html')
'''