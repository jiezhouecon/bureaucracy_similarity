# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name: 
# Purpose:
#
# Author: JieZhou
#
# Created: 5/18/20
# Copyright: (c) JieZhou 5/18/20
# ----------------------------------------------
import datetime
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from gensim.models.wrappers import DtmModel
from similarity2 import calculator

method = "dtm_cos"
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

dir_pickle = r"dtm/{}.pkl".format(method)

try:
    with open(dir_pickle, 'rb') as file:
        sim_alldepts = pickle.load(file)
except FileNotFoundError:
    sim_alldepts = {}
    with open(dir_pickle, 'wb') as file:
        pickle.dump(sim_alldepts, file)

for code1, code2 in depts:
    # if code1 != code2:
    try:
        sim_dict = sim_alldepts[(code1, code2)]
    except KeyError:
        sim_dict = {}

    leader_list = leaders[code1]
    univ1 = code2univ[int(code1.split()[0])]
    univ2 = code2univ[int(code2.split()[0])]
    print(univ1, univ2)

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
    sim_calculator = calculator(df1, df2, abs=False, token=False)

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

    sim_calculator.prep_dtm(dic_dir='dim/model', model_dir='dim/corpus.csv')

    for i1 in leader_pubs:
        for i2 in all_pubs:
            try:
                print(i1, i2)
                # i1, i2 = (17409, 53974) '018 F'
                sim_dict[(i1, i2)] = sim_calculator.dtm_cos((i1, i2), (code1, code2))
                print(sim_dict[(i1, i2)])
            except:
                sim_dict[(i1, i2)] = np.nan

    if code1 == code2:
        for i1, i2 in pairs_sameauthor[code1]:
            if (i1, i2) not in sim_dict:
                try:
                # similarity should be the function used to generate the similarity between paper with index i1 in df1 and i2 in df2
                    sim_dict[(i1, i2)] = sim_calculator.dtm_cos((i1, i2), (code1, code2))
                except:
                    sim_dict[(i1, i2)] = np.nan

    sim_alldepts[(code1, code2)] = sim_dict
    print(code1, code2, datetime.datetime.now())

    with open(r"dtm/{}.pkl".format(method), 'wb') as file:
        pickle.dump(sim_alldepts, file)