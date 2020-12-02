import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import time

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score 
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, '../src')
import helpers as h

SEED = 123
FOLDS = 5
CPU = -1

"""
FUNCTIONS IN LIBRARY:
--------------------
cvtrain_classifier()
test_classifer()
score_classifers()
plot_cv_metrics()
plot_individual_metric()
plot_all_metrics()
"""

def cvtrain_classifier(clf, classifiers, X, y):
    """
    Runs cross validation on the pipeline stored in the classifiers dictionary under subkey['pipeline']
    using X and y data. Results are store in the classifiers dictionary. 
    Displays cross validation execution time if TimeIt=True.

    Parameters
    ----------
    clf: string
    X: pandas dataframe
    y: pandas series
    """    

    cv_scores = cross_validate(classifiers[clf]['pipeline'], X, y,
                                cv=StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED), 
                                return_train_score=False, 
                                scoring=["average_precision", "recall", "precision", "f1"])

    classifiers[clf]["cv_Avg_Precision_scores"] = cv_scores['test_average_precision']
    classifiers[clf]["cv_Recall_scores"] = cv_scores['test_recall']
    classifiers[clf]["cv_Precision_scores"] = cv_scores['test_precision']         
    classifiers[clf]["cv_f1_scores"] = cv_scores['test_f1']   
    
    # use average to calculate a singel score:
    classifiers[clf]["cvAvg_Avg_Precision_score"] = np.mean(classifiers[clf]["cv_Avg_Precision_scores"])
    classifiers[clf]["cvAvg_Recall_score"] = np.mean(classifiers[clf]["cv_Recall_scores"])
    classifiers[clf]["cvAvg_Precision_score"] = np.mean(classifiers[clf]["cv_Precision_scores"])
    classifiers[clf]["cvAvg_f1_score"] = np.mean(classifiers[clf]["cv_f1_scores"])

    return classifiers

def test_classifier(clf, classifiers, X_train, y_train, X_test, y_test):
    '''
    Fits the pipeline stored in classifiers dictionary on X_train, y_train data.
    Makes predictions with pipiline on X_test, y_test data.  
    Results are store in the classifiers dictionary.
 
    Parameters
    ----------
    clf: string
    '''     

    pipeline = classifiers[clf]['pipeline'].fit(X_train, y_train)    
       
    classifiers[clf]["y_pred"] = pipeline.predict(X_test)
    classifiers[clf]["test_Avg_Precision_score"] = average_precision_score(y_test, 
                                                               classifiers[clf]["y_pred"])
    classifiers[clf]["test_Recall_score"] = recall_score(y_test, 
                                                    classifiers[clf]["y_pred"])
    classifiers[clf]["test_Precision_score"] = precision_score(y_test, 
                                                          classifiers[clf]["y_pred"])
    classifiers[clf]["test_f1_score"] = f1_score(y_test, 
                                            classifiers[clf]["y_pred"])    
    return classifiers
                                            
def score_classifiers(clf_lst, classifiers, X_train, y_train, X_test, y_test, TimeIt=True):
    """
    Populates classifier dictionary keys with cv and test predictions
 
    Parameters
    ----------
    clfassifiers: dictionary 
    X_train:
    y_train:
    """
    for clf in clf_lst:
        start_time = time.time()

        classifiers = cvtrain_classifier(clf, classifiers, X_train, y_train) 
        classifiers = test_classifier(clf, classifiers, X_train, y_train, X_test, y_test)
        
        if TimeIt:
            t = time.time() - start_time
            print(f"{t:.0f} seconds cross_validate execution time for {clf} classifier")

def plot_cv_metrics(ax, clf_lst, classifiers):
    """
    Plots the average cross validation score for each classifer for the metrics
    recall, precision, f1, average precision

    Parameters
    ----------
    ax: plot axis
    clf_lst: list
    classifiers: dictionary
    """
    colors = []
    clf_desc = []
    for clf in clf_lst:
        colors.append(classifiers[clf]["c"])
        clf_desc.append(classifiers[clf]["clf_desc"])
        results = pd.DataFrame.from_dict(classifiers, 
                                         orient='index')[["clf_desc", 
                                                        "cvAvg_Recall_score",
                                                        "cvAvg_Precision_score",
                                                        "cvAvg_f1_score",
                                                        "cvAvg_Avg_Precision_score",]]   
    results = results.loc[clf_lst]
    results.set_index('clf_desc', inplace=True)
    
    results.columns = ["Recall", "Precision", "f1", "Avg Precision"] 
    results.T.plot(kind='bar', color=colors, alpha=0.5, rot=0, ax=ax)
    ax.set_title("Average Cross Validation Score")
    ax.set_ylim(ymin=0, ymax=1.0);
    ax.grid('on', axis='y')
    ax.legend(loc='lower center')

def plot_individual_metric(ax, clf_lst, classifiers, metric):
    y1 = []
    y2 = []
    for clf in clf_lst:
        y1.append(classifiers[clf]["test_" + metric + "_score"])
        y2.append(classifiers[clf]["cvAvg_" + metric + "_score"])
    xval = np.arange(len(clf_lst))

    ax.bar(xval, y1, alpha=0.2,zorder=1, label="test score")
    for i in range(FOLDS-1):
        ax.scatter(xval, [classifiers[clf]["cv_" + metric + "_scores"][i] for clf in clf_lst], c="grey", zorder=2)
    ax.scatter(xval, [classifiers[clf]["cv_" + metric + "_scores"][FOLDS-1] for clf in clf_lst], c="grey", zorder=2, label="cv scores")
    ax.scatter(xval, y2, c="r", zorder=3, label="mean cv score")
    ax.set_xticks(np.arange(len(clf_lst)))
    ax.set_xticklabels(clf_lst)
    ax.set_ylim(0.25, 1)
    ax.set_title(metric)
    ax.grid(axis="x")
    ax.axhline(0) 
    ax.legend(loc="lower center")
    plt.tight_layout()

def plot_all_metrics(clf_lst, classifiers):
    """
    Plots cv metrics and test metrics for each classifiers for use in evaluating
    performance and potential over-fitting
    
    Parameters
    ----------
    clf_lst: list
    classifiers: dictionary
    Returns
    -------
    fig: pyplot figure
    """
    fig, axs = plt.subplots(1, 5, figsize=(20, 6))
    plot_cv_metrics(axs[0], clf_lst, classifiers)
    plot_individual_metric(axs[1], clf_lst, classifiers, "Recall")
    plot_individual_metric(axs[2], clf_lst, classifiers, "Precision")
    plot_individual_metric(axs[3], clf_lst, classifiers, "f1")
    plot_individual_metric(axs[4], clf_lst, classifiers, "Avg_Precision")
    return fig

def create_voting_classifier(clf_lst, classifiers, X_fit, y_fit):
    """
    Instantiates and fits a soft voting classifier, VotingClassifier, that takes the 
    average probability by combining the predicted probabilities. Hard voting uses the 
    predicted class labels and takes the majority vote. Weights can be assigned to the 
    model predictions when you know one model outperfomrs the others significantly.
    
    Parameters
    ----------
    clf_lst: list
    classifiers: dictionary
    X_fit: dataframe with training data
    y_fit: data series with training class

    Returns
    -------
    fig: dictionary
    """
    estimators = []
    for clf in clf_lst:
        estimators.append([clf, classifiers[clf]['pipeline']])
    pipeline = Pipeline([['Voting', VotingClassifier(estimators, voting = 'soft')]])   
    if "Voting" not in classifiers: 
        classifiers["Voting"] = {"clf_desc": "Voting Ensemble",
                                "model": VotingClassifier(estimators, voting = 'soft'), 
                                "c": "purple", 
                                "cmap":  plt.cm.Purples,
                                "threshold": 0.5,
                                "DecisionFunction": False,
                                "pipeline": pipeline}
        start_time = time.time()
        classifiers["Voting"]['pipeline'].fit(X_fit, y_fit) 
        t = time.time() - start_time
        print(f"{t:.0f} seconds cross_validate execution time for Voting classifier")
    return classifers

if __name__ == '__main__':

   classifiers = h.load_classifier_dict("classifiers_bak.pickle")
   clf_lst = [clf for clf in classifiers]
   fig = plot_all_metrics(clf_lst, classifiers)
   fig.savefig("../images/metric_scores.png")
