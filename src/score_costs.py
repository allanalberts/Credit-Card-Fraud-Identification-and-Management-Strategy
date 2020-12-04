"""
FUNCTIONS IN LIBRARY:
--------------------
cs_confution_matrix()
full_review()
partial_review()
no_review()
class_probabilities()
plot_confusion_matrix()
plot_multiple_confusion_matrix()
plot_total_cost()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import time
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import confusion_matrix
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

FraudBudget=0.0005
ReviewCost=10
ChargebackFee=20


def cs_confusion_matrix(y_test, y_pred, cost_matrix):
    """
    Returns a cost sensitive confusion matrix using the cost matrix
 
    Parameters
    ----------
    y_test: array 
    y_pred: array
    cost_matrix: array

    Returns
    -------    
    array
    """

    cost_TN = np.sum((1 - y_test) * (1 - y_pred) * cost_matrix[:, 0])
    cost_FP = np.sum((1 - y_test) * y_pred * cost_matrix[:, 1])
    cost_FN = np.sum(y_test * (1 - y_pred) * cost_matrix[:, 2])
    cost_TP = np.sum(y_test * y_pred * cost_matrix[:, 3])

    return np.array([[cost_TN, cost_FP],[cost_FN, cost_TP]])

def full_review(c, ReviewCost, ChargebackFee):
    '''
    Returns a cost_matrix that contains, for every transaction, the four 
    possible costs associated with it depending on the outcome of its 
    classification (TN, FP, FN, TP). The model assumes we will manually 
    review all suspected fraud transactions and cancel those in which fraud 
    is confirmed.     

    Parameters
    ----------
    c: array consisting of each transactions dollar amount
    ReviewCost: float
    ChargebackFee: float

    Returns
    -------
    cost_matrix: array
    ''' 
    n_samples = c.shape[0]
    cost_matrix = np.zeros((n_samples, 4))
    cost_matrix[:, 0] = 0.0
    cost_matrix[:, 1] = ReviewCost                                                               
    cost_matrix[:, 2] = c + ChargebackFee 
    cost_matrix[:, 3] = ReviewCost
    return cost_matrix 

def partial_review(c, ReviewCost, ChargebackFee):
    '''
    Returns a cost_matrix that contains, for every transaction, the four 
    possible costs associated with it depending on the outcome of its 
    classification (TN, FP, FN, TP). The model assumes a manually review 
    suspected fraud transactions with a value less than the ReviewCost. 

    Parameters
    ----------
    c: array consisting of each transactions dollar amount
    ReviewCost: float
    ChargebackFee: float

    Returns
    -------
    cost_matrix: array
    ''' 
    n_samples = c.shape[0]
    cost_matrix = np.zeros((n_samples, 4))
    cost_matrix[:, 0] = 0.0 
    cost_matrix[:, 1] = np.where(c>=ReviewCost, ReviewCost, c*0.5)                                                               
    cost_matrix[:, 2] = c + ChargebackFee 
    cost_matrix[:, 3] = np.where(c>=ReviewCost, ReviewCost, 0)
    return cost_matrix 

def no_review(c, ChargebackFee):
    '''
    Returns a cost_matrix that contains, for every transaction, the four 
    possible costs associated with it depending on the outcome of its
    classification (TN, FP, FN, TP). The model assumes all transaction 
    suspected of fraud will be cancelled without a manual review.
 
    Parameters
    ----------
    c: array consisting of each transactions dollar amount
    ChargebackFee: float

    Returns
    -------
    cost_matrix: array
    ''' 
    n_samples = c.shape[0]
    cost_matrix = np.zeros((n_samples, 4))
    cost_matrix[:, 0] = 0.0 
    cost_matrix[:, 1] = c*0.5                                                             
    cost_matrix[:, 2] = c + ChargebackFee 
    cost_matrix[:, 3] = 0.0
    return cost_matrix 

def class_probabilities(clf_lst, classifiers, X_test, y_test, cost_matrix=None):
    '''
    Fits the pipeline stored in classifiers dictionary under subkey ['pipeline'] on X_train, y_train data.
    Calculates probabilities for each sample in data X_test. Uses the predictions to create both regular and 
    cost-sensitive confusion matrices which are then stored in the classifiers dictionary structure.
 
    Parameters
    ----------
    clf: string
    X_train, X_test: Pandas DataFrames
    y_train, y_test: Pandas Series
    cost_matrix: array
    '''
    for clf in clf_lst:
 #      pipeline = classifiers[clf]['pipeline'].fit(X_train, y_train)
        classifiers[clf]["pred_prob"] = classifiers[clf]["pipeline"].predict_proba(X_test)[:,1]
            
        # store confusion matrix values - use pred_prob since generated through cross validation        
        classifiers[clf]["cnf_matrix"] = confusion_matrix(y_test,classifiers[clf]["pred_prob"]>=classifiers[clf]["threshold"])
        if cost_matrix is not None:
            classifiers[clf]["cs_cnf_matrix"] = cs_confusion_matrix(y_test, classifiers[clf]["pred_prob"]>=classifiers[clf]["threshold"],
                                                                  cost_matrix)  
            classifiers[clf]["TotalCosts"] = classifiers[clf]["cs_cnf_matrix"].sum()

def plot_confusion_matrix(ax, cm, title, classes=['Legitimate','Fraud'],
                          cmap=plt.cm.Blues, currency=False):
    """
    Plots a single confusion matrix. If currency=True then displays results as currency.

    Parameters
    ----------
    cm: array (confusion matrix)
    title: String
    test_size: float - size/percentage of holout dataset
    goal: float - project goal for ultimate dollar loss rate

    Returns
    -------
    """   
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cost=cm[i, j]
        if currency:
            cost = f'${cost:0,.2f}' 
        ax.text(j, i, cost, horizontalalignment="center", 
        color="white" if cm[i, j] > thresh else "black")
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if currency:
        ax.set_title(f'{title}\nCost Matrix')
    else:
        ax.set_title(f'{title}\nConfusion Matrix')
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, rotation=90)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')    

def plot_multiple_confusion_matrix(clf_lst, classifiers, CostSensitive):
    '''
    Plots multiple side by side confustion matrices of the classifiers in clf_list.
    If cost_matrix included, then confusion matrix will represent costs instead of occurrances.
    '''
    plt.rcParams.update(plt.rcParamsDefault)
    fig, axs = plt.subplots(1, len(clf_lst), figsize=(16, 4))
    i = 0
    for clf in clf_lst:
        if CostSensitive:
            plot_confusion_matrix(axs[i],classifiers[clf]["cs_cnf_matrix"], clf, cmap=classifiers[clf]["cmap"], currency=True)
        else:
            plot_confusion_matrix(axs[i], classifiers[clf]["cnf_matrix"], clf, cmap=classifiers[clf]["cmap"], currency=False)
        i += 1
    plt.tight_layout()
    
    return fig

def plot_total_cost(ax, clf_lst, classifiers, c, y):
    '''
    Plots the cost metric (total operational cost of fraud management model) for 
    classifiers in clf_list. Plots the operational budget target as a dashed line on plot.
    '''
    colors = []
    clf_desc = []
    for clf in clf_lst:
        colors.append(classifiers[clf]["c"])
        clf_desc.append(classifiers[clf]["clf_desc"])
    results = pd.DataFrame.from_dict(classifiers, orient='index')[["clf_desc","TotalCosts"]]
    results = results.loc[clf_lst]
    results.set_index('clf_desc', inplace=True)
    results.T.plot(kind='bar', ax=ax, color=colors, alpha=0.5, rot=0) 
    ax.axhline(y=h.total_legit(c,y)*FraudBudget, color='black', linestyle='dashed')
    ax.set_title("Costs vs. Budget (dashed line)")
    ax.legend(loc='lower right') 

    if __name__ == '__main__':
        cost_matrix = partial_review(c_test, ReviewCost, ChargebackFee)