import pandas as pd


def add_ef_classifiers_to_dict(clf_lst, classifiers):
    for clf in clf_lst:
        ef_clf = clf + "_ef"
        if ef_clf not in classifiers:
            classifiers[ef_clf] = {}
            classifiers[ef_clf]["clf_desc"] = classifiers[clf]["clf_desc"]
            classifiers[ef_clf]["model"] = classifiers[clf]["model"]
            classifiers[ef_clf]["c"] = classifiers[clf]["c"]
            classifiers[ef_clf]["cmap"] = classifiers[clf]["cmap"]
            classifiers[ef_clf]["threshold"] = classifiers[clf]["threshold"]
            classifiers[ef_clf]['pipeline'] = classifiers[clf]['pipeline']
    return classifiers

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

