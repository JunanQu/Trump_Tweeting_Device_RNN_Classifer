import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import csv

# data = pd.read_json('data/merged.json')
data = pd.read_csv('data/train.csv')

# Keeping only the neccessary columns
# data = data[['text','source']]
data = data[['text','label']]


# data['text'] = data['text'].apply(lambda x: x.lower())
# data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

# print(data[data['source'] == "Twitter for iPhone"].size)
# print(data[data['source'] == "Twitter for Android"].size)
#
# print(data[data['label'] == 1].size)
# print(data[data['label'] == -1].size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196


# def create_model():
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(2, activation='softmax'))
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(15, activation='sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
    # return model

# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [5, 10, 15]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# model = KerasClassifier(build_fn=create_model, verbose=2)

# grid = GridSearchCV(estimator=model, param_grid=param_grid)

# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
batch_size = [10, 20, 40, 60, 80]
epochs = [10, 15]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [1, 5, 10, 15, 20, 25, 30]

param_grid = dict(batch_size=batch_size, epochs=epochs) #,optimizer=optimizer, weight_constraint = weight_constraint, dropout_rate = dropout_rate)
#model = KerasClassifier(build_fn=create_model, verbose=2)

#grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter = 3)

Y = pd.get_dummies(data['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

# grid_result = grid.fit(X_train, Y_train)
# print(grid_result.best_params_)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
quit()
batch_size = 40
epoch = 10
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)

validation_size = 100

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

test = data = pd.read_csv('data/test.csv')
test = test[['text']]

test['text'] = test['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

T = tokenizer.texts_to_sequences(test['text'].values)
T = pad_sequences(T)


res = []
for t in T:
    t = [t]
    t = pad_sequences(t, maxlen=81, dtype='int32', value=0)
    sentiment = model.predict(t, batch_size=1, verbose=2)[0]
    print(np.argmax(sentiment))
    if (np.argmax(sentiment) == 0):
        res.append(-1)
    elif (np.argmax(sentiment) == 1):
        res.append(1)

with open("data/train_40_10_relu_softmax.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in res:
        writer.writerow([val])