# -*- coding: utf-8 -*-
!pip install feature-engine

"""

# Commented out IPython magic to ensure Python compatibility.
# Import required libraries

# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from scipy import stats

from feature_engine.selection import DropDuplicateFeatures
from sklearn.feature_selection import VarianceThreshold
from imblearn.under_sampling import RandomUnderSampler
# from helper import plot_boundary


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split , cross_validate , StratifiedShuffleSplit
from sklearn.metrics import accuracy_score , confusion_matrix , f1_score , roc_curve , roc_auc_score
from sklearn.model_selection import cross_val_score , GridSearchCV
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier , AdaBoostClassifier ,GradientBoostingClassifier

import scipy.optimize as opt
import numba
from numba import *

import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train = df_train.copy()
testing = df_test.copy()

train.head()

"""We can observe that there are a lot of repeptitive varibales in the the datasets with lot of duplications. Feature scalng and is required as too many variaables in the dataset will lead to over fitting. There are a lot of columns having a contant number throughout the column which wont help us in any predictions.

Now lets have a look at our target variable " TARGET".

### Checking for balance
"""

sns.countplot(test['TARGET'])

"""### DATA PRE-PROCESSING"""

# Splitting data into 6 categories based on the first name of columns
var , imp, ind, num  , saldo , delta = [], [], [],[],[],[]

for col in train.columns:
    split = col.split('_')[0]
    if split.startswith('var') == True:
        var.append(col)
    elif split.startswith('imp') == True:
        imp.append(col)
    elif split.startswith('ind') == True:
        ind.append(col)
    elif split.startswith('num') == True:
        num.append(col)
    elif split.startswith('saldo') == True:
        saldo.append(col)
    else:
        delta.append(col)
        
var_dict = {'var_name'  : ['var','imp','ind','num','saldo','delta'],
            'var_count' : [len(var),len(imp),len(ind),len(num),len(saldo),len(delta)],
            }

plt.bar(var_dict['var_name'], var_dict['var_count'])
plt.grid()
plt.xlabel("Variable Category",fontsize = 15)
plt.ylabel("Number of Predictors",fontsize = 15)

print(pd.DataFrame(var_dict))

def info(var_list,df):
    info_var = dict()
    mean = []
    Var = []
    mode = []
    range_ = []
    for v in var_list:
        mean.append((df[v].mean()))
        Var.append(stats.tvar(df[v]))
        mode.append(stats.mode(df[v]))
        range_.append([df[v].min(), df[v].max()])
        

    info_var['Mean'] = mean
    info_var['Var'] = Var
    info_var['Mode'] = mode
    info_var['Range'] = range_
    
    return pd.DataFrame(info_var,
            index = var_list)

info(var,train)

# DEFINING A FUNCTION TO IDENTITY AND SEGREGATE DUPLICATE COLUMNS
@numba.jit
def fake_columns(var_list,df):
    dupis = DropDuplicateFeatures()
    dupis_train = dupis.fit(df[var_list])
    duplicates_train = list(dupis_train.features_to_drop_)
    return duplicates_train

# SAVING THE COUNT OF THE DUPLICATE COLUMNS IN EACH VARIABLE CATEGORY IN A DICTIONARY FOR EASY VISUALIZATION
duplicate_columns = {'Var' : len(fake_columns(var,train)),
 'imp' : len(fake_columns(imp,train)),
 'ind' : len(fake_columns(ind,train)),
 'num' : len(fake_columns(num,train)),
 'saldo' : len(fake_columns(saldo,train)),
 'delta' : len(fake_columns(delta,train))
}

plt.bar(duplicate_columns.keys(),duplicate_columns.values(),color = 'red')
plt.xlabel("Variable Category",fontsize = 15)
plt.ylabel("Number of Duplicate Predictors",fontsize = 15)

# FILTERING DUPLICATE COLUMNS
var_or = [col for col in var if col not in fake_columns(var,train)]
imp_or = [col for col in imp if col not in fake_columns(imp,train)]
ind_or = [col for col in ind if col not in fake_columns(ind,train)]
num_or = [col for col in num if col not in fake_columns(num,train)]
saldo_or = [col for col in saldo if col not in fake_columns(saldo,train)]
delta_or = [col for col in delta if col not in fake_columns(delta,train)]

var_dict['Predictor_count_after_deleting_duplicates'] = [len(var_or),len(imp_or),len(ind_or),len(num_or),len(saldo_or),len(delta_or)]

pd.DataFrame(var_dict)

"""Now that we no longer have any duplicate columns in the data , We can Look further for columns with variance as 0 which mean all the values in the column are the same,again not adding value to model. Further, we also look if there are any predictors having more than 90% of the values as 0"""

# CHECKING FOR COLUMNS WITH VARIANCE = 0 OR
# MORE THAN 90% OF ELEMENTS ARE 0 
@numba.jit
def zero_var(var_list,df,threshold):
    thresh = 0.00
    for col in var_list:
        if (stats.tvar(df[col]) == threshold) or (np.percentile(df[col],90) == 0.00):
            var_list.remove(col)
    
    return var_list

var_dict['Predictor_count_removing_zero_var/sparse'] = [len(zero_var(var_or,train,0.00)),len(zero_var(imp_or,train,0.00)),len(zero_var(ind_or,train,0.00)),len(zero_var(num_or,train,0.00)),len(zero_var(saldo_or,train,0.00)),len(zero_var(delta_or,train,0.00))]
pd.DataFrame(var_dict)

var = zero_var(var_or,train,0.00)
imp = zero_var(imp_or,train,0.00)
ind = zero_var(ind_or,train,0.00)
num = zero_var(num_or,train,0.00)
saldo = zero_var(saldo_or,train,0.00)
delta = zero_var(delta_or,train,0.00)

train = train[var+imp+num+saldo+delta+ind]
train.shape

test = testing[var+imp+num+saldo+delta+ind]
test.shape

info(var,train)

fig , ax = plt.subplots(1,4, figsize = (20,5))

for i,k in enumerate(var):
    ax[i].hist(train[var[i]])
    ax[i].set_title("Population distribution of {}".format(var[i]))

X = train.drop(['TARGET','ID'],axis=1)
Y = train['TARGET']

print( X.shape , Y.shape)

x = X.copy()
y = Y.copy()

print(x.shape , y.shape )

rs = RandomUnderSampler(random_state=2)
X, y = rs.fit_resample(x,y)
x_train_ds , x_val_ds ,y_train_ds , y_val_ds = train_test_split(X,y, test_size = 0.2 , random_state = 22)
print(x_train_ds.shape ,y_train_ds.shape)
sns.countplot(y_train_ds)
plt.title("Target Distribution After Down Sampling")

rs = RandomUnderSampler(random_state=2)
X, y = rs.fit_resample(x,y)
x_train_ds , x_val_ds ,y_train_ds , y_val_ds = train_test_split(X,y, test_size = 0.2 , random_state = 22)
print(x_train_ds.shape ,y_train_ds.shape)
sns.countplot(y_train_ds)
plt.title("Target Distribution After Down Sampling")

@numba.jit()
def lreg(x,y,max_iter,c):
    lreg = LogisticRegression(max_iter = max_iter , C = c)
    lreg.fit(x,y)
    return lreg

@numba.jit()
def cv(est,x,y,folds):
    cv = cross_validate(est,
                       x,
                       y,
                       cv = folds,
                       scoring = 'f1',
                       return_train_score = True)
    return cv['test_score'] , cv['train_score']

@numba.jit()
def bootstrap(df):
    selectionIndex = np.random.randint(len(df), size = len(df))
    new_df = df.iloc[selectionIndex]
    return new_df

@numba.jit()
def best_classifier(clf, params, X,y, n_folds = 5):
    gs = GridSearchCV(clf, param_grid = params, cv=n_folds)
    gs.fit(X,y)
    print("BEST", gs.best_params_, round(gs.best_score_,2))
    best = gs.best_estimator_
    return best

# logistic regression without hyperparameter tuning 

logreg_raw = lreg(x_train_ds, y_train_ds,1000,0.001)
pred = logreg_raw.predict(x_val_ds)
print("F1 score without hyper parameter tuning for logistic regression : ",f1_score(y_val_ds,pred))

#hyper parameter tuning for logistic regression using bootstraping and cross validation 

logreg_tun = LogisticRegression(solver= 'lbfgs', max_iter=10000)
c_values = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

best_lreg = best_classifier(logreg_tun, c_values,x_train_ds,y_train_ds, n_folds = 5)

# running model on the best paramter to roc score 
lreg_tuned = lreg(x_train_ds,y_train_ds,max_iter = 10000,c = 1)
predictions = lreg_tuned.predict(x_val_ds)

logreg_auc_raw = roc_auc_score(y_val_ds,pred)
logreg_auc_tuned = roc_auc_score(y_val_ds,predictions)

y_probs_raw = logreg_raw.predict_proba(x_val_ds)[:,1]
y_probs_tuned = lreg_tuned.predict_proba(x_val_ds)[:,1]

# your code here
plt.xkcd(randomness=0,scale=0.1)
fig, ax = plt.subplots(figsize = (7,5))
fig.patch.set_facecolor('None')
fig.patch.set_alpha(0)

fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_val_ds, y_probs_raw)
fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_val_ds, y_probs_tuned)

ax.plot(fpr_raw, tpr_raw, label=f'Without Tuning (area = {logreg_auc_raw:.2f})',color= 'r')
ax.plot(fpr_tuned, tpr_tuned, label=f'Hyperparamerter Tuned (area = {logreg_auc_tuned:.2f})',color = 'b')

ax.plot([0, 1], [0, 1],'r--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve (Logistic Regression)')
ax.legend(loc="lower right")

ax.patch.set_facecolor('None')
ax.patch.set_alpha(0)
plt.tight_layout()

@numba.jit()
def tree(x,y,depth):
    dt = DecisionTreeClassifier(max_depth = depth )
    dt.fit(x,y)
    return dt

max_depth = 10
dtree_raw = tree(x_train_ds,y_train_ds,max_depth)
pred_dt_raw = dtree_raw.predict(x_val_ds)
print("F1 Score Decision Tree without hyperparameter tuning : {}".format(f1_score(y_val_ds,pred_dt_raw)))

#hyper parameter tuning for Decision Tree using Grid Search CV 

dtree_tuned = DecisionTreeClassifier()
depth = {"max_depth": [6,25,15,30,18,12,35,40]
            }

best_dtree = best_classifier(dtree_tuned, depth,X,y, n_folds = 10)

dtree_tuned = tree(x_train_ds,y_train_ds,6)
predictions = dtree_tuned.predict(x_val_ds)

dtree_auc_raw = roc_auc_score(y_val_ds,pred_dt_raw)
dtree_auc_tuned = roc_auc_score(y_val_ds,predictions)

y_probs_raw = dtree_raw.predict_proba(x_val_ds)[:,1]
y_probs_tuned = dtree_tuned.predict_proba(x_val_ds)[:,1]

# your code here
plt.xkcd(randomness=0,scale=0.1)
fig, ax = plt.subplots(figsize = (7,5))
fig.patch.set_facecolor('None')
fig.patch.set_alpha(0)

fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_val_ds, y_probs_raw)
fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_val_ds, y_probs_tuned)

ax.plot(fpr_raw, tpr_raw, label=f'Without Tuning (area = {dtree_auc_raw:.2f})',color= 'r')
ax.plot(fpr_tuned, tpr_tuned, label=f'Hyperparamerter Tuned (area = {dtree_auc_tuned:.2f})',color = 'b')

ax.plot([0, 1], [0, 1],'r--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve (Decison Tree)')
ax.legend(loc="lower right")

ax.patch.set_facecolor('None')
ax.patch.set_alpha(0)
plt.tight_layout()

@numba.jit()
def rf(x,y,depth,n_est,alpha):
    rf = RandomForestClassifier(max_depth = depth, 
                               n_estimators = n_est,
                               ccp_alpha = alpha)
    rf.fit(x,y)
    return rf

rf_raw = rf(x_train_ds,y_train_ds,6,1000,0.001)

pred_rf_raw = rf_raw.predict(x_val_ds)
print("F1 score of model on Random Forest Without tuning : {}".format(f1_score(y_val_ds,pred_rf_raw)))

# Bagging 
@numba.jit()
def bag(x,y,depth,n_est):
    base_est_tree = DecisionTreeClassifier(max_depth=depth)
    baggin = BaggingClassifier(base_est_tree,n_estimators = n_est)
    baggin.fit(x,y)
    return baggin

baggin_tree_raw = bag(x_train_ds,y_train_ds,depth = 6,n_est = 1000)
pred_bag_raw = baggin_tree_raw.predict(x_val_ds)
print("F1 score on bagging using Decision tree using the best depth from above : ",f1_score(y_val_ds,pred_bag_raw))

def SGB(x,y,lr,n_est,):
    model = GradientBoostingClassifier(loss = 'deviance',
                                      learning_rate = lr,
                                      n_estimators = n_est)
    model.fit(x,y)
    return model

@numba.jit()
def SGB(x,y,lr,n_est,):
    model = GradientBoostingClassifier(loss = 'deviance',
                                      learning_rate = lr,
                                      n_estimators = n_est)
    model.fit(x,y)
    return model

filtered_test = test.loc[test['TARGET'].isnull() == False,:]

for i in range(len(filtered_test)):
    if filtered_test.loc[i,'TARGET'] == -1:
        filtered_test.loc[i,'TARGET'] = 0

X_test = filtered_test.drop(['TARGET','ID'],axis=1)
y_test = filtered_test['TARGET']

@numba.jit()
def SGB(x,y,lr,n_est,):
    model = GradientBoostingClassifier(loss = 'deviance',
                                      learning_rate = lr,
                                      n_estimators = n_est)
    model.fit(x,y)
    return model

sg = SGB(x_train_ds,y_train_ds,0.1,1000)

print("F1 score on test data for Logistic Regression : ",f1_score(y_test,lreg_tuned.predict(X_test)))
print("F1 score on test data for Decision Tree : ",f1_score(y_test,dtree_tuned.predict(X_test)))
print("F1 score on test data for Random Forest : ",f1_score(y_test,rf_raw.predict(X_test)))
print("F1 score on test data for Logistic Regression : ",f1_score(y_test,rf_raw.predict(X_test)))
print("F1 score on test data for Gradient Boosting : ",f1_score(y_test,sg.predict(X_test)))

"""# UN-Balanced Train Set"""

X = train.drop(['TARGET','ID'],axis=1)
Y = train['TARGET']
print( X.shape , Y.shape)

strt_split = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=22)
for train_index, test_index in strt_split.split(X,Y):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]

# logistic regression without hyperparameter tuning 
logreg_raw = lreg(X_train,y_train,1000,0.001)
pred = logreg_raw.predict(X_val)

print("F1 score without hyper parameter tuning for logistic regression ")
f1_score(y_val,pred)

#hyper parameter tuning for logistic regression using bootstraping and cross validation 
logreg_tun = LogisticRegression(solver= 'lbfgs', max_iter=10000)
c_values = {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

best_lreg = best_classifier(logreg_tun, c_values,X_train,y_train, n_folds = 5)

# running model on the best paramter to roc score 
lreg_tuned = lreg(X_train,y_train,max_iter = 10000,c = 0.0001)
predictions = lreg_tuned.predict(X_val)

logreg_auc_raw = roc_auc_score(y_val,pred)
logreg_auc_tuned = roc_auc_score(y_val,predictions)

y_probs_raw = logreg_raw.predict_proba(X_val)[:,1]
y_probs_tuned = lreg_tuned.predict_proba(X_val)[:,1]

# your code here
plt.xkcd(randomness=0,scale=0.1)
fig, ax = plt.subplots(figsize = (7,5))
fig.patch.set_facecolor('None')
fig.patch.set_alpha(0)

fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_val, y_probs_raw)
fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_val, y_probs_tuned)

ax.plot(fpr_raw, tpr_raw, label=f'Without Tuning (area = {logreg_auc_raw:.2f})',color= 'r')
ax.plot(fpr_tuned, tpr_tuned, label=f'Hyperparamerter Tuned (area = {logreg_auc_tuned:.2f})',color = 'b')

ax.plot([0, 1], [0, 1],'r--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve (Logistic Regression)')
ax.legend(loc="lower right")

ax.patch.set_facecolor('None')
ax.patch.set_alpha(0)
plt.tight_layout()

@numba.jit()
def tree(x,y,depth):
    dt = DecisionTreeClassifier(max_depth = depth )
    dt.fit(x,y)
    return dt

max_depth = 10
dtree_raw = tree(X_train,y_train,max_depth)
pred_dt_raw = dtree_raw.predict(X_val)
print("F1 Score Decision Tree without hyperparameter tuning : {}".format(f1_score(y_val,pred_dt_raw)))

#hyper parameter tuning for Decision Tree using Grid Search CV 

dtree_tuned = DecisionTreeClassifier()
depth = {"max_depth": [6,25,15,30,18,12,35,40]
            }

best_dtree = best_classifier(dtree_tuned, depth,X,Y, n_folds = 10)

dtree_tuned = tree(X_train,y_train,6)
predictions = dtree_tuned.predict(X_val)

dtree_auc_raw = roc_auc_score(y_val,pred_dt_raw)
dtree_auc_tuned = roc_auc_score(y_val,predictions)

y_probs_raw = dtree_raw.predict_proba(X_val)[:,1]
y_probs_tuned = dtree_tuned.predict_proba(X_val)[:,1]

# your code here
plt.xkcd(randomness=0,scale=0.1)
fig, ax = plt.subplots(figsize = (7,5))
fig.patch.set_facecolor('None')
fig.patch.set_alpha(0)

fpr_raw, tpr_raw, thresholds_raw = roc_curve(y_val, y_probs_raw)
fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_val, y_probs_tuned)

ax.plot(fpr_raw, tpr_raw, label=f'Without Tuning (area = {dtree_auc_raw:.2f})',color= 'r')
ax.plot(fpr_tuned, tpr_tuned, label=f'Hyperparamerter Tuned (area = {dtree_auc_tuned:.2f})',color = 'b')

ax.plot([0, 1], [0, 1],'r--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve (Decison Tree)')
ax.legend(loc="lower right")

ax.patch.set_facecolor('None')
ax.patch.set_alpha(0)
plt.tight_layout()

@numba.jit()
def rf(x,y,depth,n_est,alpha):
    rf = RandomForestClassifier(max_depth = depth, 
                               n_estimators = n_est,
                               ccp_alpha = alpha)
    rf.fit(x,y)
    return rf

rf_raw = rf(X_train,y_train,6,1000,0.001)

pred_rf_raw = rf_raw.predict(X_val)
print("F1 score of model on Random Forest Without tuning : {}".format(f1_score(y_val,pred_rf_raw)))

@numba.jit()
def SGB(x,y,lr,n_est,):
    model = GradientBoostingClassifier(loss = 'deviance',
                                      learning_rate = lr,
                                      n_estimators = n_est)
    model.fit(x,y)
    return model

sg = SGB(X_train,y_train,0.1,1000)

print("F1 score on test data for Logistic Regression : ",f1_score(y_test,lreg_tuned.predict(X_test)))
print("F1 score on test data for Decision Tree : ",f1_score(y_test,dtree_tuned.predict(X_test)))
print("F1 score on test data for Random Forest : ",f1_score(y_test,rf_raw.predict(X_test)))
print("F1 score on test data for Logistic Regression : ",f1_score(y_test,rf_raw.predict(X_test)))
print("F1 score on test data for Gradient Boosting : ",f1_score(y_test,sg.predict(X_test)))

