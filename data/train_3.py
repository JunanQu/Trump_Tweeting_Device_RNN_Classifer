import pandas as pd
import nltk
import pickle
import random
import csv
from nltk.corpus import stopwords

custom_stopwords = ['crookedhillary', 'kaine', 'trump2016', 'trumppence2016', 'votetrump', 'iacaucus', 'tedcruz', 'ted',
                    'cruz', 'bernie', 'sanders', 'trumppence16', 'jeb', 'bush', 'lyin', 'gopdebate', "lindsey",
                    "graham"]

# Read tweets and label with "trump" and "staff"
tweets = pd.read_json("data/condensed_2016.json")
tweets['year'] = 2016
for year in [2017]:
    temptweets = pd.read_json("data/condensed_" + str(year) + ".json")
    temptweets['year'] = year
    tweets = tweets.append(temptweets)

for year in [2015]:
    temptweets = pd.read_json("data/condensed_" + str(year) + ".json")
    temptweets['year'] = year
    tweets = tweets.append(temptweets)

for x in [2021]:
    temptweets = pd.read_json("data/train.json")
    if temptweets['label'].all() == -1:
        temptweets['source'] = 'Twitter for Android'
    tweets = tweets.append(temptweets)

tweets['type'] = ['1' if source == 'Twitter for Android' else '-1' for source in tweets['source']]

# Condense tweets down to simplest components
train_tweets = []
test_tweets = []
for (index, row) in tweets.iterrows():
    if row.is_retweet == False:
        # test_tweets.append((nltk.word_tokenize(row['text'].lower()), row['type']))
        train_tweets.append((nltk.word_tokenize(row['text'].lower()), row['type']))

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
# test_set = nltk.classify.apply_features(extract_features, test_tweets)

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
# print(nltk.classify.accuracy(classifier, test_set))

real_test = pd.read_json("data/test.json")
r_t = []
for(index, row) in real_test.iterrows():
    r_t.append(( nltk.word_tokenize(row['text'].lower())))

rt = nltk.classify.apply_features(extract_features, r_t)

results = classifier.classify_many([fs for (fs,l) in rt])

with open("data/res.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in results:
        writer.writerow([val])
print(classifier.show_most_informative_features(20))