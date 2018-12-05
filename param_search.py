import numpy as np
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
import csv
from keras.optimizers import Adam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh
from keras.activations import relu, elu, tanh, sigmoid
from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer
import talos as ta


data = pd.read_json('./data/train.csv')
data = data[['text','label']]

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


def create_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Embedding(max_fatures, params['embed_dim'],input_length = x_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(params['lstm_out'], dropout=params['dropout'], recurrent_dropout=params['dropout']))

    # model.add(Dense(12, input_dim=8, activation='relu'))
    # model.add(Dense(2, activation='softmax'))

    model.add(Dense(params['first_neuron'], input_dim=8, activation=params['activation']))
    model.add(Dense(y_train.shape[1], activation = params['last_activation']))
    model.compile(loss = params['loss'], optimizer=params['optimizer'],metrics = ['accuracy'])
#     model.fit(X_train, Y_train, epochs = params['epochs'], batch_size=params['batch_size'], verbose = 2)
    
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=early_stopper(params['epochs'], mode='strict'))
    
    print(model.summary())
    
    return out, model
  
p = {
     'first_neuron':[8, 16, 24, 32],
     'embed_dim' : [32,64, 96, 128],
     'lstm_out' : [49,98, 147 ,196],
     'hidden_layers':[0, 1, 2],
     'batch_size': (5, 10, 15, 20, 40),
     'epochs': [7, 10, 15],
     'dropout': [0.1 ,0.2,0.3, 0.9],
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick'],
     'optimizer': ['adam'],
     'loss': [categorical_crossentropy],
     'activation' : [relu],
     'last_activation': [softmax]
}


param_grid = p

# model = KerasClassifier(build_fn=create_model, verbose=2)


# grid = GridSearchCV(estimator=model, param_grid=param_grid)

Y = pd.get_dummies(data['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


h = ta.Scan(X_train, Y_train, params=p,
            model=create_model,
            dataset_name='first_try',
            experiment_no='1',
            grid_downsample=0.0006,
            search_method = "random")


e = ta.Evaluate(h)
e.evaluate(X_test, Y_test, average='macro')

print(h.details)
#r = ta.Reporting('experiment_log.csv')
#r.evaluate(X_test, Y_test, folds=10, average='macro')
#r.plot_hist()
#print(r.high('val_fmeasure'))
#print(r.best_params())
r = ta.Reporting(h)
print(r.high())
best_params = r.best_params()
print("best:", best_params[0:min(len(best_params),10)])
print("worst:", best_params[len(best_params)-3:])

d = ta.Deploy(h, "model_1")

# grid_result = grid.fit(X_train, Y_train)
# print(grid_result.best_params_)
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# quit()

'''
batch_size = 40
epoch = 10

validation_size = 100

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

test = data = pd.read_csv('./data/test.csv')
test = test[['text']]

test['text'] = test['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

T = tokenizer.texts_to_sequences(test['text'].values)
T = pad_sequences(T)


res = []
for t in T:
    t = [t]
    t = pad_sequences(t, maxlen=33, dtype='int32', value=0)
    sentiment = model.predict(t, batch_size=1, verbose=2)[0]
    print(np.argmax(sentiment))
    if (np.argmax(sentiment) == 0):
        res.append(-1)
    elif (np.argmax(sentiment) == 1):
        res.append(1)

with open("./data/train_40_10_relu_softmax.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in res:
        writer.writerow([val])
'''
