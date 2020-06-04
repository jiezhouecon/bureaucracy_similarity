# -*- coding: utf-8 -*-
# ---------------------------------------------
# Name:
# Purpose: Train TF-IDF model
#
# Author: JieZhou
#
# Created: 3/24/20
# Copyright: (c) JieZhou 3/24/20
# ----------------------------------------------
from gensim.models import TfidfModel

from compile_data import compile2
from preprocess import prepare_lda

train = compile2()  # using the whole corpus as the train set instead of department-level corpus

train.insert(0, 'id', range(0, len(train)))
texts, doc_vectors, dictionary = prepare_lda(train)
dictionary.save('model/tfidf.dict')

model = TfidfModel(doc_vectors)

model.save('model/tfidf')