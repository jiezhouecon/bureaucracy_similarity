# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 3/28/20
# Copyright: (c) JieZhou 3/28/20
# ----------------------------------------------
import re
import pickle
import pandas as pd
from preprocess import prepare_lda
import lda


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
    train = train.append(df[df.index.isin(all_pubs)], sort=False)

'''
texts, doc_vectors, dictionary = prepare_lda(train)
# dictionary.save('ldaModel/dict_t100.dict')

lda_model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
lda_model.fit(doc_vectors)
lda_model.save('ldaModel/lda_t100')

'''
for year in range(2000, 2020):
    train_year = train[year + 3 >= train['Year']]
    # train_year = train_year.sample(n=size)
    print(len(train_year))
    texts, doc_vectors, dictionary = prepare_lda(train_year)
    dictionary.save('ldaModel/dict_t100_y%d.dict' % year)

    lda_model = lda.LDA(n_topics=100, n_iter=15000, random_state=1)
    lda_model.fit(doc_vectors)
    lda_model.save('ldaModel/lda_t100_y%d' % year)

    #visualisation = pyLDAvis.gensim.prepare(lda_model, doc_vectors, dictionary=dictionary)
    #pyLDAvis.save_html(visualisation, 'ldaModel/visLDA/lda_test.html' % year)


