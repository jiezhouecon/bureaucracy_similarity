# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 4/20/20
# Copyright: (c) JieZhou 4/20/20
# ----------------------------------------------
# !/usr/local/bin/python
import pickle
import datetime
import pandas as pd
from similarity2 import calculator

method = "doc2vec_100_cos"
# Please change this directory accordingly
with open("leaders.pkl", "rb") as file:
    leaders = pickle.load(file)

# Please change this directory accordingly
with open("code2univ.pkl", "rb") as file:
    code2univ = pickle.load(file)

with open("department_list.pkl", "rb") as file:
    depts = pickle.load(file)

with open(r"pairs_sameauthor.pkl", "rb") as file:
    pairs_sameauthor = pickle.load(file)

dir_pickle = r"doc2vec/{}.pkl".format(method)

try:
    with open(dir_pickle, 'rb') as file:
        sim_alldepts = pickle.load(file)
except FileNotFoundError:
    sim_alldepts = {}
    with open(dir_pickle, 'wb') as file:
        pickle.dump(sim_alldepts, file)

for code1, code2 in depts:
    try:
        sim_dict = sim_alldepts[(code1, code2)]
    except KeyError:
        sim_dict = {}

    leader_list = leaders[code1]
    univ1 = code2univ[int(code1.split()[0])]
    univ2 = code2univ[int(code2.split()[0])]

    # Please change this directory accordingly
    df1 = pd.read_csv(r'Merged/' + univ1 + '.csv')
    df2 = pd.read_csv(r'Merged/' + univ2 + '.csv')

    # Please change this directory accordingly
    with open(r"Paper List/" + code1 + '.pkl',
              "rb") as file:
        pubs1 = pickle.load(file)

    with open(r"Paper List/" + code2 + '.pkl',
              "rb") as file:
        pubs2 = pickle.load(file)

    # Initialize the similarity calculator
    sim_calculator = calculator(df1, df2)

    leader_pubs = []
    for leader in leader_list:
        if leader in pubs1:
            leader_pubs.extend(pubs1[leader])

    all_pubs = []
    for faculty in pubs2:
        all_pubs.extend(pubs2[faculty])

    all_pubs = list(set(all_pubs))
    leader_pubs = list(set(leader_pubs))
    print("pubs length: %d" % len(all_pubs))
    print("leader pubs length: %d" % len(leader_pubs))

    sim_calculator.prep_doc2vec()

    for i1 in leader_pubs:
        for i2 in all_pubs:
            print(i1, i2)
            # similarity should be the function used to generate the similarity between paper with index i1 and i2 in df
            sim_dict[(i1, i2)] = sim_calculator.doc2vec((i1, i2))
            print(sim_dict[(i1, i2)])

    if code1 == code2:
        for i1, i2 in pairs_sameauthor[code1]:
            if (i1, i2) not in sim_dict:
                # similarity should be the function used to generate the similarity between paper with index i1 in df1 and i2 in df2
                sim_dict[(i1, i2)] = sim_calculator.doc2vec((i1, i2))

    sim_alldepts[(code1, code2)] = sim_dict

    print(code1, code2, datetime.datetime.now())

    with open(r"doc2vec/{}.pkl".format(method), 'wb') as file:
        pickle.dump(sim_alldepts, file)