# Delayma: A Comprehensive Approach to Accurate Flight Delay Forecasting

### Authors:
- [Anirudh S. Kumar](https://github.com/Anirudh-S-Kumar)
- [Prakhar Gupta](https://github.com/Prakhar-Gupta-03)
- [Sahil Saraswat](https://github.com/sahilence)
- [Vartika](https://github.com/Vartika2401)

## Motivation
Flight delays have long been a centre of uncertainty in the aviation industry, having widespread effects on the aviation industry and passengers.It is a critical concern within the aviation industry, impacting passenger satisfaction, operational costs, and overall system efficiency. Prediction of delay can have far-reaching consequences on the entire industry and is of huge economic importance. 

Accurately predicting flight delays is a crucial point of research as it is affected by various factors-weather, air traffic congestion, logistic complexities, festivities, and economics. Developing reliable prediction models can revolutionize how airlines manage their schedules and passengers plan their journeys. This study aimed at creating a modern ML model which can accurately predict delays based on- season, time, location, etc. For this study, we used various datasets and intend to employ relevant ML algorithms to correctly predict delays. 

## Related Work

## Dataset Details
The dataset source is from [this kaggle dataset](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations), which contains information about flights that took off from JFK Airport between November 2019 and January 2020.

Datasets used for Classification had their `DEP_DELAY` column converted to binary classes based on delay, where delay is true if the departure time delay exceeds 15 minutes.

To simplify the classification problem we decided to begin with the smaller problem and then proceed to the bigger one. To proceed, four sub-datasets were created from this.
- **df_1_3** - Dataset with top 3 most frequent destinations.
- **df_1_10** - Dataset with top 10 most frequent destinations.
- **df_1_25** - Dataset with top 25 most frequent destinations.
- **df_preprocessed** - Dataset with all destinations.

For the regression problem, the DEP_DELAY column was used as it is.
- **m1_final_regression_dataset** - Dataset with all destinations for regression.



## Data Pre-processing
Basic preprocessing, which was done on the complete dataset:-
    
The feature initially had 25 different conditions. In some data points, the Condition field had more than one condition. First, we performed an encoding similar to label encoding in a monotonically increasing way. For example, there were different kinds of cloudy, such as 'mostly cloudy' and 'partly cloudy', for which we gave numbers 2 and 1, respectively. Each feature was given one column, which resulted in 9 new columns and the deletion of the Condition column.

Parameters like day, month, hour and minute are repetitive. Cyclic feature engineering is required on such cyclic parameters, where we represent one parameter on a circle, with all the points on the circle showing its periodic properties. Therefore, cyclic feature engineering was done on columns MONTH, DAY_OF_WEEK, DAY_OF_MONTH.  

## Methodology
### Baseline for classification
We have reproduced results from the papers and used the algorithms they have used in the papers to set a baseline from the previous studies. On our preprocessed data, we now use the Synthetic Minority Oversampling Technique(SMOTE), which uses KNN as its hidden layer algorithm to synthesize samples of minority classes to deal with the class imbalance(quite severe in our dataset). After we've dealt with class imbalance, we perform an 80:20 split and scale the data. Following this, we use the Boruta algorithm to select features, a complex algorithm involving Random Forests to automate feature selection. Random forest model produced the best scores given below:
![Baseline Results](images/baseline_performance.png)

### Classification 
We used mutiple algorithms for multiple datasets and compared their performance.
#### Classification for df_1_3
We used the following algorithms for this dataset: <br>
Logistic regression <br>
Hyperparameters :
```
{Penalty = l2 Tolerance = 1e-05 Max Iterations = 500 Solver = lbfgs}
```
Bayesian classifier <br>
Hyperparameters :
```
{Alpha = 0.1}
```
Passive Aggressive Classifier <br>
Hyperparameters :
```
{default}
```
SGD Classifier  <br>
Hyperparameters :
```
{Default}
```
Among these algorithms, Logistic regression performed the best with an accuracy of 0.96 and an F1 score of 0.83.

#### Classification for df_1_10
We used the following algorithms for this dataset: <br>
Logistic regression <br>
Random Forest Classifier <br>

We used GridSearchCV to find the best parameters for the Random Forest Classifier and Logistic regression. The best parameters for Random Forest Classifier were:
```
{'max_depth': 4, 'max_features': 'log2', 'n_estimators': 100}
```
The best parameters for Logistic regression were:
```
{'C': 0.01, max_iter': 1000}
```

#### Classification for df_1_25
We used the following algorithms for this dataset:
Random Forest Classifier <br>
Hyperparameters :
```
{n_estimators = 400}
```
XGBoost Classifier <br>
Hyperparameters :
```
{colsample_bytree=1.0, gamma=0, max_depth=5, min_child_weight=5, subsample=1}
```
LightGBM Classifier <br>
Hyperparameters :
```
{num_leaves = 100}
```
CatBoost Classifier <br>
Hyperparameters :
```
{depth = 5,iterations = 1000,learning_rate = 0.1}
```

## Results
The result using multiple algorithms for multiple datasets are as follows:
![Classification Results](images/model_performances.png)


We trained the model on complete datset using these algorithms with same hyperparameters and got the following results:
![Classification Results](images/results_on_final_data.png)

## Observations

## Conclusion and Future Work

## References

