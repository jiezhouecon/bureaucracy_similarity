# -*- coding: utf-8 -*-
# !/usr/local/bin/python
# ---------------------------------------------
# Name:
# Purpose:
#
# Author: JieZhou
#
# Created: 4/14/20
# Copyright: (c) JieZhou 4/14/20
# ----------------------------------------------
import pickle
import datetime
import pandas as pd
from similarity2 import calculator

method = "tfidf_jsd"
# Please change this directory accordingly
with open("leaders.pkl", "rb") as file:
    leaders = pickle.load(file)

# Please change this directory accordingly
with open("code2univ.pkl", "rb") as file:
    code2univ = pickle.load(file)

with open("department_list.pkl", "rb") as file:
    depts = pickle.load(file)

try:
    with open(r"result/{}.pkl".format(method), 'rb') as file:
        sim_alldepts = pickle.load(file)
except FileNotFoundError:
    sim_alldepts = {}
    with open(r"result/{}.pkl".format(method), 'wb') as file:
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

    sim_calculator.prep_tfidf('tfidf/tfidf.dict', 'tfidf/tfidf')

    for i1 in leader_pubs:
        for i2 in all_pubs:
            print(i1, i2)
            sim_dict[(i1, i2)] = sim_calculator.tfidf((i1, i2))
            print(sim_dict[(i1, i2)])

    sim_alldepts[(code1, code2)] = sim_dict
    print(code1, code2, datetime.datetime.now())

    with open(r"tfidf/{}.pkl".format(method), 'wb') as file:
        pickle.dump(sim_alldepts, file)