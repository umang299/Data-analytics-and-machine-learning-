# Data-analytics-and-machine-learning-
This repo contains projects on data analytics and machine learning. The libraries i have used here are pandas , numpy , matplotlib, seaborn and sklearn , scipy 

# Machine learning life cycle 

### Introduction 
* The dataset i have used is Kaggle california housing prices [ https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-California-Housing-Prices/master/Data/housing.csv ] 
* Task is to predcit the **median housing price** values in different parts of california based on various other parameters.
* This is an example of **supervised batch learning multivariate regression** problem since the target variable is a continuous numeric value.   

### Selecting a perfromance metric 
* Since the feature data contains outliers its better to choose a metric and model that is less sensitive to outliers and doesnot overfit leading to poor performane on test data 
* Here i have used **mean_absolute_perentage_error** as a metric to evalute the performace (https://i.imgur.com/ndIXERr.jpg) 
