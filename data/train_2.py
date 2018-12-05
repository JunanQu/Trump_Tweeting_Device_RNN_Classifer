import pandas as pd
import nltk
import pickle
import random
import numpy as np
import csv
from nltk.corpus import stopwords

custom_stopwords = ['crookedhillary', 'kaine', 'trump2016', 'trumppence2016', 'votetrump', 'iacaucus', 'tedcruz', 'ted',
                    'cruz', 'bernie', 'sanders', 'trumppence16', 'jeb', 'bush', 'lyin', 'gopdebate', "lindsey",
                    "graham"]

# Read tweets and label with "trump" and "staff"
tweets = pd.read_csv("data/train.csv")
print(tweets)

temptweets = pd.read_json("data/test.json")

temptweets['usage'] = "test"
tweets = tweets.append(temptweets)

# tweets['type'] = ['1' if source =='Twitter for Android' else 'staff' for source in tweets['source']]
tweets['type'] = ['1' if source == 1 else '-1' for source in tweets['label']]

# Condense tweets down to simplest components
train_tweets = []
test_tweets = []

for (index, row) in tweets.iterrows():
    # print(type(row.isRetweet))
    # if row.isRetweet:
    # print(row)
    if row['usage'] == "test":
        test_tweets.append((nltk.word_tokenize(row['text'].lower()), row['type']))
    else:
        train_tweets.append((nltk.word_tokenize(row['text'].lower()), row['type']))

print(len(test_tweets))

# Remove stopwords
train_tweets = [
    ([word for word in tweet[0] if word not in stopwords.words('english') and word not in custom_stopwords], tweet[1])
    for tweet in train_tweets]


def get_word_features(tweets):
    all_words = []
    for (words, sentiments) in tweets:
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
classifier = nltk.NaiveBayesClassifier.train(training_set)

# print("saving classifier")
# file = open('classify/trump_classifier.pickle', 'wb')
# pickle.dump(classifier, file, -1)
# file.close()
# file = open('classify/trump_classifier_features.pickle', 'wb')
# pickle.dump(word_features, file, -1)
# file.close()

print("accuracy test")
print()
print(nltk.classify.accuracy(classifier, test_set))

k = 0
res = []
for row in test_set:
    res.append(row[-1])
    k + 1

with open("data/res.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in res:
        writer.writerow([val])