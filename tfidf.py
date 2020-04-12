# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 3/24/20
# Copyright: (c) JieZhou 3/24/20
# ----------------------------------------------
import pickle
import re
import datetime
import pandas as pd
from gensim.matutils import cossim
from gensim.models import TfidfModel

from preprocess import prepare_lda


def sim(model, corpus, indices):
    v1 = model[corpus[indices[0]]]
    v2 = model[corpus[indices[1]]]
    sim = cossim(v1, v2)
    return sim

with open("leaders.pkl", "rb") as file:
    leaders = pickle.load(file)

with open("code2univ.pkl", "rb") as file:
    code2univ = pickle.load(file)

for code in dict((k, v) for k, v in leaders.items() if k in list(leaders.keys())[1:]):
    train = pd.DataFrame(columns=['ccode', 'pcode', 'Title', 'Year', 'Authors', 'Affiliations',
                                  'Journal', 'Keywords', 'Abstract', 'Link', 'WFID', 'Leader'])
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

    train.insert(0, 'id', range(0, len(train)))
    texts, doc_vectors, dictionary = prepare_lda(train)
    model = TfidfModel(doc_vectors)

    sim_dict = {}
    for i1 in leader_pubs:
        for i2 in all_pubs:
            print((i1, i2))
            j1 = train.loc[i1, 'id']
            j2 = train.loc[i2, 'id']
            # similarity should be the function used to generate the similarity between paper with index i1 and i2 in df
            sim_dict[(i1, i2)] = sim(model, doc_vectors, (j1, j2))
            print(sim_dict[(i1, i2)])

    with open('tfidf/' + ccode + '.pkl', 'wb') as file:
        pickle.dump(sim_dict, file)
        print(code, len(leader_pubs), len(all_pubs), datetime.datetime.now())