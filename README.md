# Trump-Twitting-Device-Classifier

## Feature Selection
A lot of the features such as latitude & longitude were just columns of NAs, so these had to be removed.


Certain features such as as id, tweet_id & retweetCount will not help discriminate the labels, so they were removed as well.


The date-time information was extensive, but not all of it is useful independently. To compensate, we extracted the day and hour and added these to our data set.


The text content was pretty important, so we used a Sentiment Analysis Package (VADER Sentiment from NLTK) to get a composite score for each tweet as a feature in the new data set. Our expectation was that the sentiment of the messages would be closely related to the label.

## Baseline Models
#### Simple model - 
For our simplest model, we implemented a Naive Bayes classifier for Trump’s tweet text (one of the better ways to classify text). This gave us a baseline accuracy of 58%.


#### Intermediate model - 
To improve on the poor performance of the Naive Bayes classifier, we decided to use a Random Forest since it is more robust to outliers. This was a great improvement, leading to an improved accuracy of 70%.


#### Advanced models - 
While individual models were not able to perform better than 0.71, we decided to use an ensemble to get more power out of our classifier. Our initial ensemble was a mix of Random Forest (RF) & Logistic Regression (LR). We added LR because it does not make a lot of assumptions about the data. We used the product of the label and the probability of prediction from each classifier (sklearn.classifier.predict_proba) for the ensemble learner. In the end, we took the sign of the prediction from this. For example, for a particular example, if RF output a value of +0.9 for a positive output (+1) and LR output +0.8 for the same, then the overall prediction would be sign(0.9 + 0.7) = sign(1.6) = +1. This makes sense, since both RF & LR are pretty sure about their individual predictions, so the overall classifier should reflect this. While this is expected, the classifier also does well on tricky predictions. For example, if RF’s value is +0.6 while LR’s value is -0.8 (a case where the RF is less confident about it being positive, while LR is more confident about it being negative), the end prediction would be sign(0.6 - 0.8) = sign(-0.2) = -1. This “weighting” reflects how we as humans make decisions as well and is much better than predicting a random label if we were simply using majority voting. This ensemble method leads to a marginal improvement - 76%

## Final System

Neural Network - Sentiment Analysis
It is intuitive to think that Trump and his staff speak in different tone, and we confirmed this idea by reading through a few tweets and finding a quite obvious cut-off in terms of tone. Most of Trump’s tweets are simply “Trumpier”. What if our model is able to catch such sentimental difference and make predictions based on that? 

We applied tokenization, text-to-sequence and word embedding techniques in order to help the model understand the context. However, it is not enough to have the model just understanding every tokenized word. Instead, the model needs to comprehend a sequence of word. Therefore, we used Recurrent Neural Network because RNN is able to connect previous information to the present task. However, one big shortcoming of RNN is its long-term dependency that RNN cannot really learn long-term information. Therefore, we picked LSTM, introduced by Hochreiter. 

When building the network, we split our training dataset into train and validation into 80% and 20%. At the same time, we kept a validation set from test to measure score and accuracy. We tried combination of “relu” and “softmax” , “tanh” and “softmax”, and also “softmax” alone while always keeping softmax as the last-activation function because our network uses categorical cross-entropy loss. Through empirical experiments, we decided to use “relu” and “softmax”. The last step for our network was parameter search. In order to find the optimal hyperparameters for our neural networks, we used a combination of grid search, randomized search, and manual search. Our results are quite lengthy, so we include them in a separate document: hyperparameter_search. There are way more searches done, and this document is just a snapshot. 

## Voting
After we submitted results from several models, we stored all results and used their Kaggle score as weights. Then, for every testing datapoint, we weigh its predicted label from each model by its Kaggle score and we find make new prediction based on the voted label. 
