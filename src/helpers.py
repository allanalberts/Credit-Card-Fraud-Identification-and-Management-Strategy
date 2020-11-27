import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED=123

def feature_engineering(data):
    """
    Returns dataframe and feature list with features that are 
    relevant only for predicting fraud <= $1 converted
    to zero (sample mean) for amounts > $1. Likewise, features
    relevant for only predicting fraud above $1 have their
    non relevant transactions converted to zero as well.

    Parameters
    ----------
    data: dataframe containing transaction dataset

    Returns
    -------
    df - Pandas Dataframe
    feature_lst - list
    """

    df = data.copy()
    df['LowAmount'] = (data['Amount'] <= 1)
    df['LowAmount'] = df['LowAmount'].astype(int)
    df['NonLowAmount'] = (data['Amount'] > 1)
    df['NonLowAmount'] = df['NonLowAmount'].astype(int)

    df['V15'] = df['LowAmount'] * data['V15']
    df['V24'] = df['LowAmount'] * data['V24']
    df['V26'] = df['LowAmount'] * data['V26']

    df['V8'] = df['NonLowAmount'] * data['V8']
    df['V19'] = df['NonLowAmount'] * data['V19']
    df['V21'] = df['NonLowAmount'] * data['V21']

    feature_lst = ['V%d' % number for number in range(1, 13)] + \
                  ['V%d' % number for number in range(14, 22)] + \
                  ['V24', 'V26']

    return df, feature_lst


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
        data, features = feature_engineering(data)
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
                                          stratify=y, shuffle=True,
                                          random_state=123)

    return X_cvtrain, X_test, \
           y_cvtrain, y_test, \
           c_cvtrain, c_test, \
           X_holdout, y_holdout, c_holdout, \
           features