"""
FUNCTIONS IN LIBRARY:
--------------------
    cvtrain_classifiers()
    fit_classifiers()
    score_classifiers()
    plot_cv_metrics()
    plot_individual_metric()
    plot_all_metrics() 
"""
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
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, '../src')
import helpers as h

SEED = 123
FOLDS = 5
CPU = -1


def cvtrain_classifiers(clf_lst, classifiers, X, y, TimeIt=True):
    """
    Runs cross validation on the pipeline stored in the classifiers dictionary under subkey['pipeline']
    using X and y data. Results are store in the classifiers dictionary. 
    Displays cross validation execution time if TimeIt=True.

    Parameters
    ----------
    clf_lst: list of classifier keys
    X: matrix to fit on
    y: array to fit on
    """    
    for clf in clf_lst:
        start_time = time.time()
        cv_scores = cross_validate(classifiers[clf]['pipeline'], X, y,
                                    cv=StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED), 
                                    return_train_score=False, 
                                    scoring=["recall", "precision", "f1"])

        classifiers[clf]["cv_Recall_scores"] = cv_scores['test_recall']
        classifiers[clf]["cv_Precision_scores"] = cv_scores['test_precision']         
        classifiers[clf]["cv_f1_scores"] = cv_scores['test_f1']   
    
        # use average to calculate a singel score:
        classifiers[clf]["cvAvg_Recall_score"] = np.mean(classifiers[clf]["cv_Recall_scores"])
        classifiers[clf]["cvAvg_Precision_score"] = np.mean(classifiers[clf]["cv_Precision_scores"])
        classifiers[clf]["cvAvg_f1_score"] = np.mean(classifiers[clf]["cv_f1_scores"])
        if TimeIt:
            t = time.time() - start_time
            print(f"{t:.0f} seconds cvfit execution time for {clf} classifier")

    return classifiers

def fit_classifiers(clf_lst, classifiers, X, y, SMOTE=False, TimeIt=True):
    """
    Return classifier dictionary with element 'fitted_model' containing the 
    classifer fitted to X, y

    Parameters
    ----------
    clf: string, the classifer
    classifiers: dictionary holding each classifiers attributes
    X: matrix to fit on
    y: array to fit on

    Returns
    -------
    classifiers: dictionary
    """
    for clf in clf_lst:
        start_time = time.time()
        if SMOTE:
            classifiers[clf]['fitted_model'] = classifiers[clf]['pipeline'].fit(X, y)      
        else: 
            classifiers[clf]['fitted_model'] = classifiers[clf]['model'].fit(X, y) 
        if TimeIt:
            t = time.time() - start_time
            print(f"{t:.0f} seconds fit execution time for {clf} classifier")
    return classifiers

def score_classifiers(clf_lst, classifiers, X, y):
    """
    Makes predictions with pipiline on X, y data. Usually test or 
    holdout data. Results are store in the classifiers dictionary.
 
     Parameters
    ----------
    clf_lst: list of classifier keys
    classifiers: dictionary holding each classifiers attributes
    X: matrix to fit on
    y: array to fit on

    Returns
    -------
    classifiers: dictionary
    """
    for clf in clf_lst:
        classifiers[clf]["y_pred"] = classifiers[clf]["fitted_model"].predict(X)
        classifiers[clf]["test_Recall_score"] = recall_score(y, 
                                                    classifiers[clf]["y_pred"])
        classifiers[clf]["test_Precision_score"] = precision_score(y, 
                                                          classifiers[clf]["y_pred"])
        classifiers[clf]["test_f1_score"] = f1_score(y, 
                                            classifiers[clf]["y_pred"])    
    return classifiers
                                            
def plot_clf_metrics(ax, clf_lst, classifiers, score_type):
    """
    Plots the average cross validation score for each classifer for the metrics
    recall, precision, f1

    Parameters
    ----------
    ax: plot axis
    clf_lst: list
    classifiers: dictionary
    """
    recall = "Recall_score"
    precision = "Precision_score"
    f1 = "f1_score"
    title = ""
    if score_type == "CV":
        recall = "cvAvg_" + recall
        precision = "cvAvg_" + precision
        f1 = "cvAvg_" + f1
        title = "Average SMOTE Upsampled Score"
    if score_type == "test":
        recall = "test_" + recall
        precision = "test_" + precision
        f1 = "test_" + f1
        title = "Average Cross Validation Score"

    colors = []
    clf_desc = []
    for clf in clf_lst:
        colors.append(classifiers[clf]["c"])
        clf_desc.append(classifiers[clf]["clf_desc"])
    results = pd.DataFrame.from_dict(classifiers, 
                                    orient='index')[["clf_desc", 
                                                    recall,
                                                    precision,
                                                    f1]]   
    results = results.loc[clf_lst]
    results.set_index('clf_desc', inplace=True)
    
    results.columns = ["Recall", "Precision", "f1"] 
    results.T.plot(kind='bar', color=colors, alpha=0.5, rot=0, ax=ax)
    ax.set_title(title)
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
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    plot_clf_metrics(axs[0], clf_lst, classifiers, score_type="CV")
    plot_individual_metric(axs[1], clf_lst, classifiers, "Recall")
    plot_individual_metric(axs[2], clf_lst, classifiers, "Precision")
    plot_individual_metric(axs[3], clf_lst, classifiers, "f1")
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
    return classifiers

if __name__ == '__main__':
    # X_train, X_test, \
    # y_train, y_test, \
    # c_train, c_test, \
    # X_holdout, y_holdout, c_holdout, \
    # features = h.load_data(4, engineered_features=False)
    # classifiers = h.initialize_classifier_dict()
    # clf_lst = [clf for clf in classifiers]

    # score_classifiers(clf_lst, classifiers, X_train, y_train, X_test, y_test, TimeIt=True)
    # h.save_classifier_dict(classifiers, "10")
    classifiers = h.load_classifier_dict('classifiers_ver10.pickle')
    clf_lst = [clf for clf in classifiers]
    fig = plot_all_metrics(clf_lst, classifiers)
    fig.savefig("../images/metric_scores.png")
