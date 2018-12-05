import pandas as pd
import nltk
import pickle
import random
import numpy as np
import csv

from keras.preprocessing.sequence import pad_sequences
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
import sklearn

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)

tweets = pd.read_json("data/train.json")
train_tweets = []

for(index, row) in tweets.iterrows():
    train_tweets.append(( nltk.word_tokenize(row['text'].lower())))

def build_indices(train_set):
    tokens = [token for line in train_set for token in line]

    # From token to its index
    forward_dict = {'UNK': 0}

    # From index to token
    backward_dict = {0: 'UNK'}
    i = 1
    for token in tokens:
        if token not in forward_dict:
            forward_dict[token] = i
            backward_dict[i] = token
            i += 1
    return forward_dict, backward_dict

def encode(data, forward_dict):
    return [list(map(lambda t: forward_dict.get(t,0), line)) for line in data]

forward_dict, backward_dict = build_indices(train_tweets)

train_inputs = encode(train_tweets, forward_dict)

padded = pad_sequences(train_inputs)




label = pd.read_csv("data/train.csv")
label['type'] = ['1' if source == 1 else '-1' for source in tweets['label']]

# for i in range(len(padded)):
#     padded[i][0]=label['type'][i]
print(padded)

with open("data/seq.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in padded:
        writer.writerow([val])