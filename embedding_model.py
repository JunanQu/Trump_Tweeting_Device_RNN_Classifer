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

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(7)

max_review_length = 500
top_words = 10000

data_path = 'data/train.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data).astype(str)

x_train = data[:,1][:900]
x_test = data[:,1][901:]
print(x_train)
y_train = data[:,17][:900]
y_test =  data[:,17][901:]

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

forward_dict, backward_dict = build_indices(x_train)
x_train = encode(x_train, forward_dict)
x_test = encode(x_test, forward_dict)
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)



# tweets = pd.read_json("data/train.json")
# tweets['usage'] = "train"
#
# tweets['type'] = ['1' if source == 1 else '-1' for source in tweets['label']]
#
# label = pd.read_csv("data/train.csv")
# label['type'] = ['1' if source == 1 else '-1' for source in tweets['label']]
#
# # Condense tweets down to simplest components
# train_tweets = []
# test_tweets = []
#
# print(train_tweets)
#
# for(index, row) in tweets.iterrows():
#     train_tweets.append(( nltk.word_tokenize(row['text'].lower()), row['type']))

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

data_path = 'data/test.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    test = list(reader)
    # transform data into numpy array
    test = np.array(data).astype(str)

forward_dict, backward_dict = build_indices(test)
test = encode(test, forward_dict)
test = sequence.pad_sequences(test, maxlen=max_review_length)
res = model.predict_classes(test, batch_size=64,verbose=0)

print(res)