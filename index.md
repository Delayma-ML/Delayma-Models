# Delayma

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
The dataset source is from a Kaggle Dataset(add dataset link), which contains information about flights that took off from JFK Airport between November 2019 and January 2020. 

To simplify the problem we decided to begin with the smaller problem and then proceed to the bigger one. To proceed, four sub-datasets were created from this.

Datasets used for Classification had their DEP_DELAY column converted to binary classes based on delay, where delay is true if the departure time delay exceeds 15 minutes.

## Data Pre-processing
Basic preprocessing, which was done on the complete dataset:-
    
The feature initially had 25 different conditions. In some data points, the Condition field had more than one condition. First, we performed an encoding similar to label encoding in a monotonically increasing way. For example, there were different kinds of cloudy, such as 'mostly cloudy' and 'partly cloudy', for which we gave numbers 2 and 1, respectively. Each feature was given one column, which resulted in 9 new columns and the deletion of the Condition column.

Parameters like day, month, hour and minute are repetitive. Cyclic feature engineering is required on such cyclic parameters, where we represent one parameter on a circle, with all the points on the circle showing its periodic properties. Therefore, cyclic feature engineering was done on columns MONTH, DAY_OF_WEEK, DAY_OF_MONTH.  
## Methodology

## Results

## Observations

## Conclusion and Future Work

## References

