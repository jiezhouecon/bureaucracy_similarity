# bureaucracy_similarity
Version control for similarity measures: [google spreadsheet](https://docs.google.com/spreadsheets/d/1okaSgrF8otUqJFWOtA0KGQf6xKsMt3GUVxFO8Ff8Ht4/edit#gid=0).

The code file `similarity.py` contrains functions for calculating several similarity measures. 
You may download this in your working directory and import it in the follwoing way:
```
from similarity import calculator
```

# 1. Initialize the similarity calculator
The code is written as a class which uses a dataframe as its object. The dataframe will be a table in which each observation is a paper. You can initialize the similarity calculator for the whole dataframe using the following code:
```
# Initialize the similarity calculator
sim_calculator = similarity(df)
```

# 2. BERT + Cosine
## Preparation
Steps:
1. Please download the model used for Chinese BERT through [this link](https://www.dropbox.com/sh/ruz967qrqujgzr4/AADOXeDvMKVpmqbNaN1HkmmQa?dl=0)
2. Install the required package for BERT and call the corresponding service. All these can be done with the follwoing code:
```
sim_calculator.prep_bert("model/chinese_L-12_H-768_A-12", wait_time = 60) 
# Please change the path to the model correspondingly
```
Notice that to make sure the model is fully loaded, I make wait_time = 60. You may change this based on your situation.

## Calculate the similarity
The similiarity score will be calculated at the pair level. The input should be a tupe of two indices. For example:
```
sim_calculator.bert_cos((9, 8))
```
This will return the BERT similarity score for paper 9 and paper 8.

You may do a simple test before running your own code:
```
from similarity import calculator
df = pd.read_csv('南开大学.csv')
df = df[df['Affiliations'].str.contains("经济")].reset_index(drop=True)
df = df.head(20)

# Initialize the similarity calculator
sim_calculator = calculator(df)
# Prepare the corresponding model for the measure
sim_calculator.prep_bert("model/chinese_L-12_H-768_A-12", wait_time = 60)
# Calculate similarity
print(sim_calculator.bert_cos((9, 8)))
```

# 3. Doc2Vec
## Preparation
Steps:
1. Please download the most updated Doc2Vec model through [this link](https://www.dropbox.com/s/b4sjx117ew6291z/d2v.model?dl=0)
2. Install the required package for Doc2Vec: `gensim 3.8.1`. (I guess you already have this installed. So I don't force this in the code. You need to manually install it if you don't have.)
3. Load the model and get tokenized texts. All these can be done with the follwoing code:
```
sim_calculator.prep_doc2vec("model/d2v.model") 
# Please change the path to the model correspondingly
```

## Calculate the similarity
The similiarity score will be calculated at the pair level. The input should be a tupe of two indices. For example:
```
sim_calculator.doc2vec((9, 8))
```
This will return the Doc2Vec similarity score for paper 9 and paper 8.

## 

# 4. LDA
## Preparation
Steps:
1. Model for LDA can be downloaded from [this link](https://www.dropbox.com/sh/f6v8fnw2j1xr3rx/AAA_Gl49gtDSdkkO_csRqiw6a?dl=0). 

There are two type of models: `vi` and `gibb`. Models in the dropbox folder `lda/vi` is trained by `train/lda.py`. And those in the dropbox folder `lda/gibb` is trained by `train/lda_gibb.py`.

2. Load the model and get tokenized texts. All these can be done with the follwoing code:
```
sim_calculator.prep_lda('ldaModel/vi/dict_t100.dict', 'ldaModel/vi/lda_t100')
# Please change the path to the dictionary and model correspondingly
```

## Calculate the similarity
The similiarity score will be calculated at the pair level. The input should be a tupe of two indices. For example:
```
sim_calculator.lda((9, 8))
```
This will return the LDA similarity score for paper 9 and paper 8.
