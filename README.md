# Credit Card Fraud Identification and Management Strategy

## Table of Contents

1. [Fraud Managent](#fraud-management-background)
2. [Business Requirements](#business-requirments)
3. [Data Source](#about-the-data)
5. [Training and Evaluation Strategy](#training-and-evaluation-strategy)
6. [Classifier Tuning](#classifier-tuning-using-cross-validation)
7. [Dealing Imbalanced Data](#dealingimbalanced-data)
8. [Production Results](production-results)
9. [PCA Visualization of Fraud](#pca-compressed-visualization-of-fraud)
10. [Recommendations](#recommendations)

## Fraud Management Background

A typical fraud management model will identify suspected fraud transactions and place them in a manual review queue, where investigators will utilize additional resources to verify the legitimacy of these transaction. Variations of this model may include automatically accepting certain transactions as well as automatically cancelling others that are suspected of being fruadulent. 

Successful fraud identification models involve limiting misclassifications: 
- False negatives or not identfying fraud. This is measured by a recall score representing the percentage of fraud captured by the model. 
- False positive or incorreclty identifying legitimate transactions as fraudulent. This is measured with precision score representing the percentage of the manual review queue that is actually fraud as opposed to a misclassified legitimate sale. 

However, measuring the effectiveness of a fraud identification model is more complex than these classification metrics as all transactions share the same misclassification costs. It is imperative that Recall of a fraud model captures high dollar fraud. However, the importance of Precision should not be underestimated, as legitimate transactions flagged as fraudulent can lead to significant lost revenue, customer satisfaction, and operational expenses associated with manual review and intervention. 

An **Example Dependent Cost-Sensitive Prediction Model** is one that is measured based on the true costs of misclassification as opposed to just the frequency of these incidents. 

### Business Requirements:
Create a business strategy for a cost efficient fraud management model that includes populating a manual review queue of suspected fraud transaction, determining the manual transaction review costs for these transactions, and calculating the losses from undetected fraud. The model must provide a wholistic fraud solution covering operational costs as well as undetected fraud losses and fall within an allocated an operational budget for fraud **FraudBudget** of 5 basis points (0.05%) of sales revenue.

### Business Rules

The fraud prediction model will output a queue of suspected fraud transactions that will then be manually reviewed/investigated to confirm the legitimacy of the suspected fraud. The average operational cost per transaction review **ReviewCost** is determined by the business to be 10 dollars. The business model will assume that, through the manual review process, the legitimacy of the suspected fraud transaction can be determined and these transactions will be cancelled. Undetected fraud transactions will result in a chargeback for the amount of the transaction plus a chargeback processing fee **ChargebackFee** assessed by the financial institution in the amount of 20 dollars per transaction. Legitimate transactions that are misidentified as fraud and cancelled will be assigned a cost equivelent to their lost revenue. 
- Fraud Budget: 5bp (0.05% of sales)
- Chargeback Fee: 20.00
- Avg Manual Review Cost: 10.00
- Will automatically decline suspected fraud when transaction amount < review costs
![](/images/cost_matrix2.png)
These potential costs are then written to a cost matrix that has the same number of rows as our dataset. The cost matrix is then applied to the predicted results using the following logic:

![](/images/cost_matrix_formulas.png)

### About the Data 

Source: https://www.kaggle.com/mlg-ulb/creditcardfraud#creditcard.csv
The datasets is provided by Kaggle and contains credit card transactions made over a two day period in September 2013 by european cardholders. It contains 492 fraudulent transactions out of 284,807 total transactions. The dataset is highly unbalanced with the positive class (frauds) accounting for 0.172% of all transactions. The dataset contains 28 transactions attributes that can be used as fraud prediction features. These features ('V1','V2',....,'V28') are the result of Principal Component Analysis (PCA) transformation which has anomynized the original features and provided confidentiality for the dataset.

## Training and Evaluation Strategy
In a real world scenario we would train on historical data and test on new transactions, however with only 48 hours of data we will hold out 20% of the data to represent new data for measuring our system when it is put into production. We will ensure that this data is equilent to our training data by taking a stratified, shuffled sample and the usage of a random state of 4 on our train_test_split will provide for an equal balance of fraud incident rate (0.00172) as well as actual fraud loss amount rate (0.00238 vs 0.00243). Having the a similar proportion of fraud transactions and net amount enabling equivelent measures of fraud per sales basis point (our requirements).
![](/images/random_state_analysis.png)

### Classifier Tuning using cross validation
The training data will then be split again so that 80% can be used for cross validation tuning and 20% for testing for overfitting prior to releasing the model for production. **Stratification** will be used with both cross_val_predict and cross_val_score. This is important because our data is highly imbalanced and without stratification there will be folds with varying proportions of fraud that will lead to very different prediction scores between the folds that won't generalize to future transactional data. **Shuffling** is also importing to use since similar fraud transactions will sometimes occur heavier in a small timeframe and our data is ordered by time.


### Classifier Selection
Multiple forms of ensemble learning will be used in the fraud detection model. 

**Bagging Based Ensemble Learning** uses multiple random models with their individual projections combined to to produce a final output. Bagging generally improves stability and accuracy. A **Random Forrest Classifier** will be used that uses multiple decision trees built on randomly selected data. Decision trees often perform well on imbalanced datasets because the splitting rules that look at the class variable used in the creation of the trees can force both classes to be addressed. 

**Boosting Based Ensemble Learning** is a sequential learning ensemble technique. Weak learners are combined to make a stong learner. The entire dataset is used to train the model and then subsequent models are built by fitting the residual error values of the initial model. Boosting attempts to give a higher weight to those observations that were poorly estimated in the previous model. Once the sequence of models is created, their preditions are weighted by their accuracy scores and the results are combined to create a final estimation. **Extreme Gradient Boosting (XGBoost)** performs boosting using decision trees.

## Classifier Evalution:
For a benchmark, our three classifiers have been cv trained and tested using their default paramters. Also shown are the test scores for each metric along with the values from each cross validation fold. Note that none of the classifiers appear to overfit the data with cross val training scores being in line with the test data score.
![](/images/metric_scores.png)

## Dealing Unbalanced Data:
Unbalanced data means that there are fewer records that you are training to predict than the ones you are not trying to predict. This causes issue for generalizing classifiers to unseen data because there aren't enough records in the minority class to effectivley learn the decision boundaries between the minority and majority records. There are two ways you can try to better balance your dataset.

### Undersampling
Undersampling is a technique where records from the majority class are removed in order to make the two classes more equal. Our data is far too imbalanced to use this approach without a large information loss.

### Synthetic Minority Oversampling Technique (SMOTE)
 Because our dataset is so highly embalanced with very few fraud records, We will use Synthetic Minority Oversampling Technique (SMOTE) to create new minority class records by taking a random near minority set neighbor of each minority record. A vector is then calculted that runs through the current record and it's selected neighbor. This vector is multiplied by a random number between zero and one and added to the current data point to create the new synthetic datapoint. 
![](/images/SMOTE_viz.png)

Below are the results of a cross validation test using various ratios of synthetically created fraud. You can see that all classifiers benefited from the additional records with their optimum ratio for improving recall scoring is noted on each plot. 

![](/images/smote_parms.png)

### SMOTE issues
If not executed correctly, SMOTE may not retain the integrity of the class distribution of the original dataset. The process must be performed during, and **not** before the cross validation process otherwise if may be validating against a copy of itself:

![](/images/smote-cross1.png)
![](/images/smote-cross2.png)

This means that a grid search for identifying optimum classifier parameters when using SMOTE must regenerate the systhetic records during each fold. The additional level of computation, makes hyperparameter tuning computationaly cost prohibitive when using SMOTE data. We will therefore only have the option of using non-SMOTE hyperparameter tuned classifiers or those tuned without synthesized data.
### SMOTE Oversampling Results
![](/images/SMOTE_score_comparison.png)

### Classifier Bagging
**Voting Based Ensemble** is fairly straight forward and it's soft setting it takes the argmax of the sums of the predicted probabilitiesas from multiple models. Individual submodels can be weighted so as to increase or decrease their impact on the final result. We will using the VotingClassifier to combine with equal weighting our strong learner classifiers that each approached the problem with a different methodology. 

### Example Dependent Cost Matrix
![](/images/test_confusion_matrix.png)
![](/images/test_confusion_matrix_cost.png)

### Anticipated Performance vs. Actual Performance
![](/images/test_total_cost.png)
![](/images/prod_total_cost.png)

![](/images/prod_confusion_matrix.png)
![](/images/prod_confusion_matrix_cost.png)


### PCA Compressed Visualization of Fraud
![](/images/PCA_visualization.png)

### Thresholding
![](/images/pr_curve.png)

Businesses often ask if the model can provide more transactions to review. This can be achieved through varying the decision threshold, however in the case of our model, the Precision-Recall Curve is aligned so that changing the recall will result in the cost of managing additional False Positives, far outweighing the benefit of additional detected fraud.

### Other Techniques - Unsupervised Anomaly Detection
Unsupervised anomaly detection models such as Isolation Forests and PCA Compression/Decompression can likely detect most of these fraud transactions that our model is missing. However, these models come at a high price for False Positives that are in the range of 20% of all transactions. This makes these models not effective from a business implementation and overall cost standpoint.

## Recommendations
Additional data will improve the model performance, however, the strongest recommendation would be to invest in new features that can be utilized to better identify the remaining undetected fraud. Further analysis will continue with the replacement of one of the tree based classifiers in this model with a neural network. In the meantime, the budget for fraud managment should be increased to 10 total basis points of sales.


