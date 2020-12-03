# Credit Card Fraud Identification and Management Strategy

## Table of Contents

1. [Fraud Managent](#fraud-management-background)
2. [Business Requirements](#business-requirments)
3. [Data Source](#about-the-data)
5. [Training and Evaluation Strategy](#training-and-evaluation-strategy)
6. [Classifier Tuning](#classifier-tuning-using-cross-validation)
7. [SMOTE](#smote)
8. [Feature Engineering](feature-engineering)
9. [Production Results](production-results)
10. [Recommendations](#recommendations)

## Fraud Management Background

A typical fraud management model will identify suspected fraud transactions and place them in a manual review queue, where investigators will utilize additional resources to verify the legitimacy of these transaction. Variations of this model may include automatically accepting certain transactions as well as automatically cancelling others that are suspected of being fruadulent. 

Successful fraud identification models involve limiting misclassifications: 
- False negatives or not identfying fraud. This is measured by a recall score representing the percentage of fraud captured by the model. 
- False positive or incorreclty identifying legitimate transactions as fraudulent. This is measured with precision score representing the percentage of the manual review queue that is actually fraud as opposed to a misclassified legitimate sale. 

However, measuring the effectiveness of a fraud identification model is more complex than these classification metrics as all transactions share the same misclassification costs. It is imperative that Recall of a fraud model captures high dollar fraud. However, the importance of Precision should not be underestimated, as legitimate transactions flagged as fraudulent can lead to significant lost revenue, customer satisfaction, and operational expenses associated with manual review and intervention. 

An **Example Dependent Cost-Sensitive Prediction Model** is one that is measured based on the true costs of misclassification as opposed to just the frequency of these incidents. 

## Business Requirements:
Create a business strategy for a cost efficient fraud management model that includes populating a manual review queue of suspected fraud transaction, determining the manual transaction review costs for these transactions, and calculating the losses from undetected fraud. The model must provide a wholistic fraud solution covering operational costs as well as undetected fraud losses and fall within an allocated an operational budget for fraud **FraudBudget** of 5 basis points (0.05%) of sales revenue.

## Business Rules

The fraud prediction model will output a queue of suspected fraud transactions that will then be manually reviewed/investigated to confirm the legitimacy of the suspected fraud. The average operational cost per transaction review **ReviewCost** is determined by the business to be 10 dollars. The business model will assume that, through the manual review process, the legitimacy of the suspected fraud transaction can be determined and these transactions will be cancelled. Undetected fraud transactions will result in a chargeback for the amount of the transaction plus a chargeback processing fee **ChargebackFee** assessed by the financial institution in the amount of 20 dollars per transaction. Legitimate transactions that are misidentified as fraud and cancelled will be assigned a cost equivelent to their lost revenue. 

## About the Data 

Source: https://www.kaggle.com/mlg-ulb/creditcardfraud#creditcard.csv
The datasets is provided by Kaggle and contains credit card transactions made over a two day period in September 2013 by european cardholders. It contains 492 fraudulent transactions out of 284,807 total transactions. The dataset is highly unbalanced with the positive class (frauds) accounting for 0.172% of all transactions. The dataset contains 28 transactions attributes that can be used as fraud prediction features. These features ('V1','V2',....,'V28') are the result of Principal Component Analysis (PCA) transformation which has anomynized the original features and provided confidentiality for the dataset.

## Training and Evaluation Strategy
In a real world scenario we would train on historical data and test on new transactions, however with only 48 hours of data we will hold out 20% of the data to represent new data for measuring our system when it is put into production. We will ensure that this data is equilent to our training data by taking a stratified, shuffled sample and the usage of a random state of 4 on our train_test_split will provide for an equal balance of fraud incident rate (0.00172) as well as actual fraud loss amount rate (0.00238 vs 0.00243). Having the a similar proportion of fraud transactions and net amount enabling equivelent measures of fraud per sales basis point (our requirements).
![](/images/random_state_analysis.png)

## Classifier Tuning using cross validation
The training data will then be split again so that 80% can be used for cross validation tuning and 20% for testing for overfitting prior to releasing the model for production. **Stratification** will be used with both cross_val_predict and cross_val_score. This is important because our data is highly imbalanced and without stratification there will be folds with varying proportions of fraud that will lead to very different prediction scores between the folds that won't generalize to future transactional data. **Shuffling** is also importing to use since similar fraud transactions will sometimes occur heavier in a small timeframe and our data is ordered by time.


## Classifier Selection
Multiple forms of ensemble learning will be used in the fraud detection model. 

**Bagging Based Ensemble Learning** uses multiple random models with their individual projections combined to to produce a final output. Bagging generally improves stability and accuracy. A Random Forrest Classifier will be used that uses multiple decision trees built on randomly selected data. Decision trees often perform well on imbalanced datasets because the splitting rules that look at the class variable used in the creation of the trees can force both classes to be addressed. 

**Boosting Based Ensemble Learning** is a sequential learning ensemble technique. Weak learners are combined to make a stong learner. The entire dataset is used to train the model and then subsequent models are built by fitting the residual error values of the initial model. Boosting attempts to give a higher weight to those observations that were poorly estimated in the previous model. Once the sequence of models is created, their preditions are weighted by their accuracy scores and the results are combined to create a final estimation. Extreme Gradient Boosting (XGBoost) performs boosting using decision trees.

**Voting Based Ensemble Learning** is fairly straight forward as it just aggregates predictions from multiple models to smooth...???????. Individual submodels can be weighted so as to increase or decrease their impact on the final result. We will using the VotingClassifier to combine with equal weighting the RandomForest and XGBoost tree based classifiers with a linear classifier that has used a completely different predictive methodology (LogisticRegression). 

## Classifier Evalution:
For a benchmark, our three classifiers have been cv trained and tested using their default paramters:
![](/images/metric scores.png)


## Example Dependent Cost Matrix

 - LogisticRegression Classifier
 - RandomForest Classifier
 - XGBoost Classifier


## Adding SMOTE
 - SMOTE
 - Pystat??

## Ensemble Classifier Bagging


