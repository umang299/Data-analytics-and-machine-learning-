#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[2]:



import pandas as pd  # for dataframe manipulation and selection 
import numpy as np  # for array operations 
from scipy.stats import chi2_contingency  # to perfrom chi-square statistic analysis 



# for data visualisation and pattern discovery 
import matplotlib.pyplot as plt  
import seaborn as sns 


from sklearn.model_selection import train_test_split , cross_val_score   # train_test_split to split the data |  corss_val_score to calculate the performance of the model 
from sklearn.impute import SimpleImputer    # impute missing values 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer ,  StandardScaler  # To scale data and convert categorical features into numeric vectors 
from sklearn.base import BaseEstimator , TransformerMixin   # Perfrom custom transfromation on the data 
from sklearn.pipeline import Pipeline, FeatureUnion         # generate pipeline 
from sklearn_features.transformers import DataFrameSelector     
from sklearn.metrics import mean_absolute_error , r2_score ,accuracy_score , mean_absolute_percentage_error # performace metrics 
from sklearn.feature_selection import SelectKBest, chi2 , f_regression    # feature selection libraries

# Importing different models 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import TheilSenRegressor


# ## Loading Dataset 

# In[3]:


def load_data(url):                                # defining a function to load dataset 
    return pd.read_csv(url)


# In[4]:


#loading dataset 
housing = load_data("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-California-Housing-Prices/master/Data/housing.csv")


# ## Data structure and description 

# In[5]:


housing.info()


# In[6]:


housing.describe()


# ## Data visualisation 

# In[7]:


bins = 50
housing.hist(bins = bins,figsize=(20,20))


# In[8]:


plt.figure(figsize=(20,20))
sns.boxplot(data = housing)


# * The features are **heavily  skewed**, measured on **scales with vast differences** and **have outliers** 

# In[9]:


label_X = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity']


# * Since the median_income is measured on a scale of 0-15 , lets create another feature as income_cat with sacle as 0-10 making easier for the model to scale and interpret 

# In[10]:


housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0,inplace = True) 
housing.info()


# ## Split the data into train, validation and test set 

# In[11]:


train_set , test_set = train_test_split(housing, test_size = 0.2 , random_state=42)
train_set, val_set = train_test_split(train_set,test_size=0.2,random_state=42)


# In[12]:


housing= train_set.copy()


# In[13]:


housing.plot(kind = 'scatter' , x = 'longitude' , y = 'latitude', alpha = 0.4,
           s = housing['population']/100 , label = 'population',
           c = 'median_house_value',cmap = plt.get_cmap('jet'),colorbar = True,
           )
plt.legend()


# In[14]:


corr_matrix = housing.corr()


# In[15]:


pd.DataFrame(corr_matrix['median_house_value'].sort_values(ascending=False))


# In[16]:


housing.plot(kind = 'scatter', x = 'median_income' , y = 'median_house_value' , alpha = 0.4)


# In[17]:


housing.columns


# In[18]:


columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'income_cat',
           'median_house_value']

categorical_attr = ['ocean_proximity']
numerical_attr = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households','income_cat',]
housing = housing[columns]


# ### Defining class for feature selection by performing statistic analysis 

# In[19]:


class feature_selection():
        def categorical_chi2(data,categorical_attr,p):
            categorical_columns = categorical_attr
            chi2_check = []
            for i in categorical_columns:
                if chi2_contingency(pd.crosstab(housing['median_house_value'], housing[i]))[1] < p:
                    chi2_check.append('Reject Null Hypothesis')
                else:
                    chi2_check.append('Fail to Reject Null Hypothesis')
            res = pd.DataFrame(data = [categorical_columns, chi2_check] 
                     ).T 
            res.columns = ['Column', 'Hypothesis']
            return res
        
        def num_selection(X_train,Y,score_func,k):
            best_features = SelectKBest(score_func=score_func,k=k)
            fit = best_features.fit(X_train,Y)
            dfscores = pd.DataFrame(fit.scores_)
            return dfscores
    


# In[20]:


feature_selection.categorical_chi2(housing,categorical_attr,p=0.05)


# * Since **ocean_proximity** rejects the null hypothesis there is no relative correlation, it can lead to bad performance of the model. Hence it is best to drop the column 

# In[21]:


housing.head()
housing = housing.drop(columns = ['ocean_proximity'], axis=1)

X = housing.iloc[:,:-1]
Y = housing.iloc[:,-1]


# ### Defining class for custom transformation

# In[22]:


room_ix , bedroom_ix , population_ix , household_ix = 3 , 4 , 5 , 6


# In[23]:


# Used to transfroms numeric features and combine numeric attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, add_bedrooms_per_room = True): # no *args / **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self    # nothing else to do 
    def transform(self,X,y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:,population_ix] / X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedroom_ix] / X[:, room_ix]
            return np.c_[X,rooms_per_household,population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,population_per_household]
    


# ### Creating data pipelines 

# In[24]:


num_pipeline=Pipeline([
    ('selector',DataFrameSelector(numerical_attr)),
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scalar',StandardScaler()),
    ])


# In[25]:


full_pipeline=FeatureUnion(transformer_list=[
    ("num_pipeline",num_pipeline),
    ])


# ### Defining class for set of models 

# In[26]:


class models():
    
    def LinearRegression(X,Y):
        model = LinearRegression()
        trained_model = model.fit(X,Y)
        return trained_model 
    
    def DesisionTreeRegressor(X,Y):
        model = DecisionTreeRegressor()
        trained_model = model.fit(X,Y)
        return trained_model
    
    def RandomForestRegressor(X,Y):
        model = RandomForestRegressor()
        trained_model = model.fit(X,Y)
        return trained_model
    
    def RansacReg(X,Y):
        model = RANSACRegressor()
        trained_model = model.fit(X,Y)
        return trained_model
        
    def TSReg(X,Y):
        model = TheilSenRegressor()
        trained_model = model.fit(X,Y)
        return trained_model
        


# In[27]:


X.head()


# ### Sending raw data through the pipeline 

# In[28]:


X_train = num_pipeline.fit_transform(X,Y)


# ### Analyse the feature importance 

# In[29]:


feature_selection.num_selection(X_train,Y, score_func = f_regression , k=7).plot(kind='bar')


# ### Evaluating the cross validation score for each model 

# In[30]:


def cross_val(model,X,Y,scoring, cv):
    scores = cross_val_score(model, X,Y,
                             scoring = str(scoring),
                             cv=10)
    scores = np.sqrt(-scores)   
    return scores


# In[31]:


def display_score(scores):
    print("Scores:",scores)
    print("mean:",scores.mean())
    print("Standard Deviation:",scores.std())


# In[32]:


lin_model=models.LinearRegression(X_train,Y)
tree = models.DesisionTreeRegressor(X_train,Y)
forest = models.RandomForestRegressor(X_train,Y)
sac = models.RansacReg(X_train,Y)


# In[33]:


scores_lr = cross_val(lin_model,X_train,Y,
         scoring = 'neg_mean_absolute_percentage_error',
         cv=10)
display_score(scores_lr)


# In[34]:


scores_tree = cross_val_score(tree, X_train,Y,
                        scoring ="neg_mean_absolute_percentage_error" ,cv=10)
scores_tree = np.sqrt(-scores_tree)
display_score(scores_tree)


# In[35]:


scores_forest = cross_val_score(forest, X_train,Y,
                      scoring = "neg_mean_absolute_percentage_error",cv=10)
scores_forest = np.sqrt(-scores_forest)
display_score(scores_forest)


# In[36]:


scores_ransac = cross_val_score(sac, X_train,Y,
                      scoring = "neg_mean_absolute_percentage_error",cv=10)
scores_ransac = np.sqrt(-scores_ransac)
display_score(scores_ransac)


# In[37]:


data = {'model':['LinearRegressiion',"DecisionTree","RandomForest","RANSACRegression"],
       'mean_error' : [scores_lr.mean(), scores_tree.mean(), scores_forest.mean(), scores_ransac.mean()],
       'Standard_Dev': [scores_lr.std(), scores_tree.std(), scores_forest.std(), scores_ransac.std()]
       }
cross_val_score = pd.DataFrame(data)
cross_val_score


# * We notice that the **RandomForestRegressor** model performs the best out of the four on the training set 

# ## Evaluting the model 

# In[38]:


val_set.drop(columns = ['ocean_proximity'],axis=1,inplace=True)


# In[39]:


val_set.columns
numerical_attr = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income','income_cat']
X = val_set[numerical_attr]
Y = val_set['median_house_value']


# In[40]:


val_x = num_pipeline.fit_transform(X,Y)


# In[41]:


pred_1 = lin_model.predict(val_x)


# In[42]:


e1 = mean_absolute_percentage_error(Y,pred_1)


# In[43]:


pred_2 = tree.predict(val_x)


# In[44]:


e2 = mean_absolute_percentage_error(Y,pred_2)


# In[45]:


pred_3 = forest.predict(val_x)


# In[46]:


e3 = mean_absolute_percentage_error(Y,pred_3)


# In[47]:


data = { "model": ['LinearRegression','DecisionTree','RandomForest'],
         "error": [ e1 , e2, e3],
       }

pd.DataFrame(data)


# ## Testing the model on test set 

# In[48]:


test_set.drop(columns = ['ocean_proximity'],axis=1,inplace=True)


# In[49]:


X = test_set[numerical_attr]
Y = test_set['median_house_value']


# In[50]:


test_X = num_pipeline.fit_transform(X,Y)


# In[51]:


final_pred = forest.predict(test_X)


# In[52]:


mean_absolute_percentage_error(Y,final_pred)


# In[ ]:




