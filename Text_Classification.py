            #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:52:10 2019

@author: girishbhatta
"""

#importing libraries
import json
import pandas as pd
import re
import numpy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier


#reading the file as a dictionary
with open("lemma_least_freq_removed.txt") as file:
    json_data = json.loads(file.read())


#------------------------------------------------------------------------------
#not required anymore    
pattern = re.compile(r"->\s?([a-zA-Z1-9]+)")
for k in json_data:
    for i in range(len(json_data[k])):
        if json_data[k][i].isalnum():
            json_data[k][i] = json_data[k][i].replace(json_data[k][i],re.search(pattern,json_data[k][i])[1].strip())
        else:
            json_data[k].remove(json_data[k][i])
#------------------------------------------------------------------------------        

# converting the dictionary to key value format and converting back to dataframe

for k in json_data:
    json_data[k] = " ".join(json_data[k])
        

tokenised_data =  pd.DataFrame(list(json_data.items()),columns=['id','review'])    
tokenised_data.head()
tokenised_data.dtypes

#reading the labels file

train_labels = pd.read_csv("train_label.csv")
train_labels.head()
train_labels.dtypes

#merging the lables with the training data
train_data = pd.merge(tokenised_data,train_labels,how = 'inner',left_on = 'id',right_on='trn_id')
train_data.head()
train_data.drop('trn_id',axis=1,inplace=True)
train_data.drop('id',axis=1,inplace = True)


# plots
Sentiment_count=train_data.groupby('label').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['review'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

# the data is equally distributed

# splitting the dataset
# naive Bayes classifier
X = train_data.review
y = train_data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

nb = Pipeline([('vect', CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,3),tokenizer = token.tokenize)),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])

    

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print('accuracy : ', metrics.accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

#### ------------------------------------------
# SGD SVM


## NOT AS GOOD AS LINEAR SVC
sgd = Pipeline([('vect', CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,3),tokenizer = token.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=1000, tol=None)),
               ])
sgd.fit(X_train, y_train)   

y_pred = sgd.predict(X_test)

print('accuracy %s' % metrics.accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))






token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,3),tokenizer = token.tokenize,
                     min_df = 0.2)
text_counts= cv.fit_transform(train_data['review'])

print(text_counts)


#splitting the dataset    
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, train_data['label'], test_size=0.3, random_state=1)


skf = StratifiedKFold(n_splits=10)
params = {}
nb = MultinomialNB()
gs = GridSearchCV(nb, cv=skf, param_grid=params, return_train_score=False)


#1 . Multinominal Naive bayes fitting and checking the model accuracy
clf = gs.fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


#using tf-idf vectorization
tf=TfidfVectorizer()
text_tf= tf.fit_transform(text_counts)

skf = StratifiedKFold(n_splits=10)
params = {}
nb = MultinomialNB()
gs = GridSearchCV(nb, cv=skf, param_grid=params, return_train_score=False)


X_train, X_test, y_train, y_test = train_test_split(
    text_tf, train_data['label'], test_size=0.3, random_state=123)

clf = gs.fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# 2. SVM Fitting and checking model accuracy


model_svc = Pipeline([('vect', CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,3),tokenizer = token.tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', LinearSVC()),
               ])
model_svc.fit(X_train, y_train)   

y_pred = model_svc.predict(X_test)

print('accuracy %s' % metrics.accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Linear SVC Accuracy:",metrics.accuracy_score(y_test, y_pred))


# 3. Logistic Regression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)





def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

