import pandas as pd
import nltk
import pickle
import random
import numpy as np
import csv
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
import sklearn

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

custom_stopwords = ['crookedhillary', 'kaine', 'trump2016', 'trumppence2016', 'votetrump', 'iacaucus', 'tedcruz', 'ted', 'cruz', 'bernie', 'sanders', 'trumppence16', 'jeb', 'bush', 'lyin', 'gopdebate', "lindsey", "graham"]

# Read tweets and label with "trump" and "staff"
# tweets = pd.read_json("data/train.json")
# tweets['type'] = ['1' if source == 1 else '-1' for source in tweets['label']]

#
tweets = pd.read_json("data/train.json")
tweets['type'] = ['1' if source == "Twitter for Android" else '-1' for source in train['label']]


test = pd.read_json("data/train.json")
test['type'] = ['1' if source == 1 else '-1' for source in test['label']]

# Condense tweets down to simplest components
train_tweets = [] 
test_tweets = []

for(index, row) in train.iterrows():
    index_of_colon = str(row["created_at"]).index(":")
    train_tweets.append(((nltk.word_tokenize(row['text'].lower()))+[(str(row["favorite_count"]))]+[(str(row["created_at"])[index_of_colon-2:index_of_colon])], row['type']))


for(index, row) in tweets_2016.iterrows():
    index_of_colon = str(row["created_at"]).index(":")
    train_tweets.append(((nltk.word_tokenize(row['text'].lower()))+[(str(row["favorite_count"]))]+[(str(row["created_at"])[index_of_colon-2:index_of_colon])], row['type']))
#
# for(index, row) in tweets.iterrows():
#     index_of_colon = str(row["created"]).index(":")
#     train_tweets.append((nltk.word_tokenize(row['text'].lower())+[(str(row["favoriteCount"]))]+[(str(row["created"])[index_of_colon-2:index_of_colon])], row['type']))

for(index, row) in test.iterrows():
    index_of_colon = str(row["created"]).index(":")
    test_tweets.append((nltk.word_tokenize(row['text'].lower())+[(str(row["favoriteCount"]))]+[(str(row["created"])[index_of_colon-2:index_of_colon])], row['type']))


for x in (train_tweets):

    for y in range(len(x[0])):
        if x[0][y] == "realdonaldtrump":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "americafirst":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "tomorrow":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "wiprimary":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "makeamericagreatagain":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "tickets":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "http":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "join":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "romney":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "rt":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "…":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "''":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "badly":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "megynkelly":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "ivankatrump":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "emails":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "teamtrump":
            x[0][y] = str(np.random.randint(0, 1000, 1))
        if x[0][y] == "0":
            x[0][y] = str(np.random.randint(0, 1000, 1))
# Remove stopwords
train_tweets = [([word for word in tweet[0] if word not in stopwords.words('english') and word not in custom_stopwords], tweet[1]) for tweet in train_tweets]

def get_word_features(tweets):
    all_words = []
    for(words, sentiments) in tweets:
        all_words.extend(words)
    
    wordlist = nltk.FreqDist(all_words)
    wordlist = wordlist.most_common()
    word_features = [word[0] for word in wordlist]
    return word_features
    
word_features = get_word_features(train_tweets)
word_features = word_features[:500]


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(' + word + ')'] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, train_tweets)

test_set = nltk.classify.apply_features(extract_features, test_tweets)

print("beginning training of trainer")
nb_classifier = nltk.NaiveBayesClassifier.train(training_set)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

# print("saving classifier")
# file = open('classify/trump_classifier.pickle', 'wb')
# pickle.dump(classifier, file, -1)
# file.close()
# file = open('classify/trump_classifier_features.pickle', 'wb')
# pickle.dump(word_features, file, -1)
# file.close()

print("accuracy test")
print(nltk.classify.accuracy(nb_classifier, test_set))




real_test = pd.read_json("data/test.json")
r_t = []
for(index, row) in real_test.iterrows():
    x = nltk.word_tokenize(row['text'].lower())
    index_of_colon = str(row["created"]).index(":")
    y = [(str(row["created"])[index_of_colon-2:index_of_colon])]
    r_t.append(x+[(str(row["favoriteCount"]))]+y)


rt = nltk.classify.apply_features(extract_features, r_t)

svc = SVC_classifier.classify_many([fs for (fs,l) in rt])

results = nb_classifier.classify_many([fs for (fs,l) in rt])
linear_svc = LinearSVC_classifier.classify_many([fs for (fs,l) in rt])

bernouli = BernoulliNB_classifier.classify_many([fs for (fs,l) in rt])


with open("data/res_2.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in results:
        writer.writerow([val])

print(nb_classifier.show_most_informative_features(30))
