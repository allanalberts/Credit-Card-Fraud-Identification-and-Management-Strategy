import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def random_state_analysis(data, test_size):
    """
    Returns 3 arrays used for random state analysis plot.
    
    Parameters
    ----------
    data: Pandas dataframe
    test_size: float - size/percentage of holout dataset

    Returns
    -------
    seed_lst: list - seed values for x axis
    train_lst: list - train dataset Fraud Dollar Loss rate values
    test_lst: list - test dataset Fraud Dollar Loss rate values
    
    """
    seed_lst = []
    train_lst = []
    test_lst = []
    for seed in range(8):
        seed_lst.append(seed)
        X = data.copy()
        y = X.pop('Class')
        X_train, X_test, \
        y_train, y_test = train_test_split(X, y,
                                    test_size=test_size,
                                    stratify=y, shuffle=True,
                                    random_state=seed)
        train_lst.append(X_train[y_train==1]['Amount'].sum() / X_train[y_train==0]['Amount'].sum() * 100)
        test_lst.append(X_test[y_test==1]['Amount'].sum() / X_test[y_test==0]['Amount'].sum() * 100)
    return seed_lst, train_lst, test_lst

def random_state_analysis_plot(ax, data , test_size, goal):
    """
    Plots Fraud Dollar Loss Rate for train and test
    datasets for different train_test_split() seed values

    Parameters
    ----------
    ax - axis to print on
    data: Pandas dataframe
    test_size: float - size/percentage of holout dataset
    goal: float - project goal for ultimate dollar loss rate

    Returns
    -------
    ax: axis with plot
    """
    seed_lst, train_lst, test_lst = random_state_analysis(data, test_size)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(seed_lst, train_lst, 'b-*', label='Fraud Dollar Loss Rate: train data')
    ax.plot(seed_lst, test_lst, 'g-*', label='Fraud Dollar Loss Rate: test data')
    ax.axhline(goal, linestyle='--', label='Fraud Dollar Loss Rate Goal')
    ax.axvline(4, linestyle='--', c="k", linewidth=0.5)
    ax.set_ylim(0)
    ax.set_xticks(seed_lst)
    ax.legend(loc="center left")
    ax.set_ylabel('Fraud Dollar Loss Rate (bp)')
    ax.set_xlabel('Random State Setting')
    ax.set_title(f'Random State Analysis for train-test ratio of {int((1 - test_size) * 100)}-{int(test_size*100)}')
    plt.tight_layout()

if __name__ == '__main__':
    data = pd.read_csv("../data/creditcard.csv")

    goal = .05
    test_size = 0.2
    fig, ax = plt.subplots(figsize=(8,4))
    random_state_analysis_plot(ax, data, test_size, goal)
    fig.savefig("../img/random_state_analysis.png")

