"""
Python version: 3.8.12
Description : Loads the chosen statistical (TF-IDF) model.
"""


# %% Importing libraries
import os
import pickle

from joblib import parallel_backend

import numpy as np

import sklearn
from sklearn.metrics import accuracy_score, make_scorer, precision_recall_fscore_support

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# %% Helper functions
def evaluateTFIDF_and_print_classification_report(trained_model, test_data, vendor_to_idx_dict):
    # Evaluating on Alpha-Dreams dataset
    vendors = [vendor if vendor in vendor_to_idx_dict.keys() else 'others' for vendor in test_data['vendor']]
    test_data['vendor'] = vendors
    test_data['vendor'] = test_data['vendor'].replace(vendor_to_idx_dict, regex=True)

    test_title = list(test_data['title'])
    test_description = list(test_data['description'])
    test_vendor = list(test_data['vendor'])

    # Merging text data
    test_corpus = ["Title : " + str(test_title[i]) + '\n' + 'Description : ' + str(test_description[i]) for i in range(test_data.shape[0])]

    predictions = trained_model.predict(test_corpus)
    # precision, recall, fscore, _ = precision_recall_fscore_support(test_vendor, predictions)
    print(sklearn.metrics.classification_report(np.array(test_vendor), np.array(predictions), digits=4))
    # return precision, recall, fscore

def evaluateTFIDF(trained_model, test_data, vendor_to_idx_dict):
    # Evaluating on Alpha-Dreams dataset
    vendors = [vendor if vendor in vendor_to_idx_dict.keys() else 'others' for vendor in test_data['vendor']]
    test_data['vendor'] = vendors
    test_data['vendor'] = test_data['vendor'].replace(vendor_to_idx_dict, regex=True)

    test_title = list(test_data['title'])
    test_description = list(test_data['description'])
    test_vendor = list(test_data['vendor'])

    # Merging text data
    test_corpus = ["Title : " + str(test_title[i]) + '\n' + 'Description : ' + str(test_description[i]) for i in range(test_data.shape[0])]

    predictions = trained_model.predict(test_corpus)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_vendor, predictions)

    return precision, recall, fscore

# %% Model Class :- TFIDF
class TFIDF(object):

    def __init__(self, train_data, test_data, stats_model_type, n_splits, directory):
        self.train_data = train_data
        self.test_data = test_data

        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        # self.scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        self.scoring = {'Accuracy': make_scorer(accuracy_score)}
        # Parameters to hypertune
        self.parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}

        self.directory = directory
        self.stats_model_type = stats_model_type

        self.train_title = list(self.train_data['title'])
        self.train_description = list(self.train_data['description'])
        self.train_vendor = list(self.train_data['vendor'])
        
        self.test_title = list(self.test_data['title'])
        self.test_description = list(self.test_data['description'])
        self.test_vendor = list(self.test_data['vendor'])

        # Merging text data
        self.train_corpus = ["Title : " + str(self.train_title[i]) + '\n' + 'Description : ' + str(self.train_description[i]) for i in range(self.train_data.shape[0])]
        self.test_corpus = ["Title : " + str(self.test_title[i]) + '\n' + 'Description : ' + str(self.test_description[i]) for i in range(self.test_data.shape[0])]

    def train_models(self):
        # TfidfTransformer is used on an existing count matrix, such as one returned by CountVectorizer
        
        # Controlling the backend that joblib will use
        with parallel_backend('threading', n_jobs=-1):
            model_dict = {}

            if self.stats_model_type == 'MultinomialNB':
                # Multinomial-NB
                clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
                gs_clf = GridSearchCV(clf, self.parameters, n_jobs=10, cv=self.cv, return_train_score=True, verbose=1, scoring=self.scoring, refit='Accuracy')
                gs_clf = gs_clf.fit(self.train_corpus, self.train_vendor)
                predictions = gs_clf.predict(self.test_corpus)
                precision, recall, fscore, support = precision_recall_fscore_support(self.test_vendor, predictions)
                pickle.dump(gs_clf, open(os.path.join(self.directory, "tfidf_trans_mnb.pickle"), 'wb'))
                model_dict['MNB'] = {'precision':np.mean(precision), 'recall':np.mean(recall), 'F1':np.mean(fscore), "model":gs_clf}

            elif self.stats_model_type == 'SVC':
                # Support Vector Machine 
                clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(class_weight='balanced', kernel="linear"))])
                gs_clf = GridSearchCV(clf, self.parameters, n_jobs=10, cv=self.cv, return_train_score=True, verbose=1, scoring=self.scoring, refit='Accuracy')
                gs_clf = gs_clf.fit(self.train_corpus, self.train_vendor)
                predictions = gs_clf.predict(self.test_corpus)
                precision, recall, fscore, support = precision_recall_fscore_support(self.test_vendor, predictions)
                pickle.dump(gs_clf, open(os.path.join(self.directory, "tfidf_trans_svm.pickle"), 'wb'))
                model_dict['SVM'] = {'precision':np.mean(precision), 'recall':np.mean(recall), 'F1':np.mean(fscore), "model":gs_clf}

            elif self.stats_model_type == 'RandomForestClassifier':
                # Random Forest
                clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=5, random_state=1))])
                gs_clf = GridSearchCV(clf, self.parameters, n_jobs=10, cv=self.cv, return_train_score=True, verbose=1, scoring=self.scoring, refit='Accuracy')
                gs_clf = gs_clf.fit(self.train_corpus, self.train_vendor)
                predictions = gs_clf.predict(self.test_corpus)
                precision, recall, fscore, support = precision_recall_fscore_support(self.test_vendor, predictions)
                pickle.dump(gs_clf, open(os.path.join(self.directory, "tfidf_trans_rf.pickle"), 'wb'))
                model_dict['RF'] = {'precision':np.mean(precision), 'recall':np.mean(recall), 'F1':np.mean(fscore), "model":gs_clf}

            elif self.stats_model_type == 'LogisticRegression':
                # Logistic Regression
                clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(class_weight='balanced', random_state=1))])
                gs_clf = GridSearchCV(clf, self.parameters, n_jobs=10, cv=self.cv, return_train_score=True, verbose=1, scoring=self.scoring, refit='Accuracy')
                gs_clf = gs_clf.fit(self.train_corpus, self.train_vendor)
                predictions = gs_clf.predict(self.test_corpus)
                precision, recall, fscore, support = precision_recall_fscore_support(self.test_vendor, predictions)
                pickle.dump(gs_clf, open(os.path.join(self.directory, "tfidf_trans_lr.pickle"), 'wb'))
                model_dict['LR'] = {'precision':np.mean(precision), 'recall':np.mean(recall), 'F1':np.mean(fscore), "model":gs_clf}

            elif self.stats_model_type == 'MLPClassifier':
                # Multilayer Perceptron Model
                clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MLPClassifier(hidden_layer_sizes=(100,100,100), random_state=1))])
                gs_clf = GridSearchCV(clf, self.parameters, n_jobs=10, cv=self.cv, return_train_score=True, verbose=1, scoring=self.scoring, refit='Accuracy')
                gs_clf = gs_clf.fit(self.train_corpus, self.train_vendor)
                predictions = gs_clf.predict(self.test_corpus)
                precision, recall, fscore, support = precision_recall_fscore_support(self.test_vendor, predictions)
                pickle.dump(gs_clf, open(os.path.join(self.directory, "tfidf_trans_mlp.pickle"), 'wb'))
                model_dict['MLP'] = {'precision':np.mean(precision), 'recall':np.mean(recall), 'F1':np.mean(fscore), "model":gs_clf}

            else:
                raise Exception("--stats_model_type can only be one amongst MultinomialNB, MLPClassifier, LogisticRegression, RandomForestClassifier, and SVC")

        return model_dict
