# Data-analytics-and-machine-learning-
This repo contains projects on data analytics and machine learning. The libraries i have used here are pandas , numpy , matplotlib, seaborn and sklearn , scipy 

# Machine learning life cycle 

### Introduction 
* The dataset i have used is Kaggle california housing prices [ https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-California-Housing-Prices/master/Data/housing.csv ] 
* Task is to predcit the **median housing price** values in different parts of california based on various other parameters.
* This is an example of **supervised batch learning multivariate regression** problem since the target variable is a continuous numeric value.   

### Selecting a perfromance metric 
* Since the feature data contains outliers its better to choose a metric and model that is less sensitive to outliers and doesnot overfit leading to poor performane on test data 
* Here i have used **mean_absolute_perentage_error** as a metric to evalute the performace ![alt text](https://i.imgur.com/ndIXERr.jpg "Logo Title Text 1")

### Discover and Visualise insights 
* Below is the histogram viualisation of numeric features
  ![alt text](https://github.com/umang299/Data-analytics-and-machine-learning-/blob/main/hist_cycle.JPG "Logo Title Text 1")
  ![alt_text](https://github.com/umang299/Data-analytics-and-machine-learning-/blob/main/box_cycle.JPG "Logo Title Text 1")
* As we observe in the boxplot, there are outliers present in the features. Hence we choose to pick a model which is not susecptible to outliers 
### Spit into train, test and validation set 
* Since the dataset is not large enough i split the data randomly using **train_test_split** with 20% test_size and random_state = 42 
### Checking for correlation 
* View the correlation between the features and targets using **.corr()** function. 

### Defining classes to automate operations on features 
* **Feature_Selection** : To perform statistical analysis on the features and pick features based on p value
* **CombinedAttribtesAdder** : To combine features and create new features with higher correlation to the target value 
* **Models** : Define various suitable models within this class 
* **cross_val** :  Perform cross validation on models and evaluate their performance on the data 

### Create Pipelines 
* Defining pipeline to perform necessary operaions on the features before feeding it to the model. 

### Finally evaluating and testing the model 
