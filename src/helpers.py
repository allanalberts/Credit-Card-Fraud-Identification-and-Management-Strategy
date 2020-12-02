import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, '../src')
import feature_engineering as ef

SEED = 123
CPU = -1
holdoutseed = 4
FOLDS = 5



def load_data(holdoutseed, engineered_features=False):
    """
    Returns datasets for cv training, testing, and a holdout
    set for final performance analysis.  Also return list containing feature 
    column names. If engineered_features is True then sprecific features
    will be modified to reflect engineered values. 
 
    Parameters
    ----------
    holdoutseed: int
    engineered_features: bool

    Returns
    -------
    X_cvtrain, X_test, X_holdout: Pandas DataFrames
    y_cvtrain, y_test, y_holdout: Pandas Series
    c_cvtrain, c_test, c_holdout: Pandas Series
    features: list
    """

    data = pd.read_csv("../data/creditcard.csv")

    if engineered_features:
        data, features = ef.feature_engineering(data)
    else:
        features = ['V%d' % number for number in range(1, 29)]
    
    X = data[features]
    y = data['Class']
    c = data['Amount']   
     
     # Create hold-out data set for final testing
    X_train, X_holdout, \
    y_train, y_holdout, \
    c_train, c_holdout = train_test_split(X, y, c, 
                                          test_size=0.20,
                                          stratify=y, shuffle=True,
                                          random_state=holdoutseed)

     # Create cv training data set and test dataset
    X_cvtrain, X_test, \
    y_cvtrain, y_test, \
    c_cvtrain, c_test = train_test_split(X_train, y_train, c_train, 
                                          test_size=0.20,
                                          stratify=y_train, shuffle=True,
                                          random_state=123)

    return X_cvtrain, X_test, \
           y_cvtrain, y_test, \
           c_cvtrain, c_test, \
           X_holdout, y_holdout, c_holdout, \
           features

def initialize_classifier_dict():
    """
    Returns a dictionary of classifier keys include colors for graphing
    Dictionary will later be populated with predictions and hyperparameters.

    clf_desc - description of classifier
    model - the instantiated and subsequently fit classifer
    pipeline - pipeline containing preprocessing steps and the instantiated classifier
    c - plot color
    cmap - confusion matrix color
    theshold - threshold value used in calculating perfomance metrics
    pred_prob - the classifiers probabilities
    PR_AUC_score - area under the PR curve
    PR_AUC_cv_scores - score for each fold
    recall_score - percentage of fraud transactions detected
    recall_cv_scores - score for each fold
    precision_score - percentage of predictions that are correct
    precision_cv_scores - score for each fold
    cnf_matrix - confusion matrix with transaction counts by results type (TN, FP, FN, TP)
    cs_cnf_matrix - cost sensitive confusion with costs associated with each results type
    TotalCosts - total operational and fraud costs of model
    -------
    classifiers: dictionary
    """

    classifiers = { 
            "RF":  {"clf_desc": "RandomForest",
                    "model": RandomForestClassifier(n_jobs=CPU, 
                                                    random_state=SEED), 
                    "c": "g", 
                    "cmap": plt.cm.Greens,
                    "threshold": 0.5}

            ,"XGB": {"clf_desc": "XGBoost",
                    "model": XGBClassifier(n_jobs=CPU, random_state=SEED), 
                    "c": "blue", 
                    "cmap": plt.cm.Blues, 
                    "threshold": 0.5}
        
            ,"LR":  {"clf_desc": "LogisticRegression",
                    "model": LogisticRegression(n_jobs=CPU, 
                                              random_state = SEED), 
                    "c": "r", 
                    "cmap": plt.cm.Reds,
                    "threshold": 0.5}
            }
    
    # Make classifier pipelines:
    for clf in classifiers:
        steps = [# placeholder for smote: ('Preprocess', FeaturePreprocess), 
                 ('Classifier', classifiers[clf]['model'])]
        classifiers[clf]['pipeline'] = Pipeline(steps)    
    
    return classifiers
    
def save_classifier_dict(classifiers, version):
    '''
    Saves the classifier dictionary in a local pickle file.
 
    Parameters
    ----------
    classifiers: dictionary
    version: string

    Returns
    -------
    None
    ''' 
    
    filename = f'classifiers_ver{version}.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)    

def load_classifier_dict(filename):
    '''
    Returns the classifier dictionary from a local pickle file where it was
    previously saved.
 
    Parameters
    ----------
    filename: string

    Returns
    -------
    classifiers: dictionary
    '''              
    with open(filename, 'rb') as handle:
        classifiers = pickle.load(handle)
    return classifiers

def total_fraud(c, y):
    TotalFraud = c.loc[y==1].sum()
    return TotalFraud
    
def total_legit(c, y):
    return c.loc[y==0].sum()


if __name__ == '__main__':

    X_train, X_test, \
    y_train, y_test, \
    c_train, c_test, \
    X_holdout, y_holdout, c_holdout, \
    features = load_data(4, engineered_features=False)

    classifiers = initialize_classifier_dict()
    clf_lst = [clf for clf in classifiers]

# score_classifiers(clf_lst)

# fig, ax = plt.subplots()
# ax = plot_classifier_metrics(ax, clf_lst)
# target_names = ['Legitimate', 'Fraud']


