#!/usr/bin/env python
# coding: utf-8

# # HOUSING ASSIGNMENT

# #PROBLEM STATEMENT#

# A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The data is provided in the CSV file below.
# 
#  
# 
# The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.
# 
#  
# 
# The company wants to know:
# 
# Which variables are significant in predicting the price of a house, and
# 
# How well those variables describe the price of a house.

# In[9]:


#importing libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[10]:


#IMPORTING THE REQUIRED DATAFRAME

housing = pd.read_csv('train.csv')


# In[11]:


housing.head()


# In[12]:


housing.shape


# In[13]:


housing.info()


# In[18]:


housing.describe().T


# In[ ]:


housing.size


# In[15]:


housing.dtypes


# In[16]:


housing.columns


# In[ ]:


# There are 1460 rows and 81 columns and all are non nulls.


# In[ ]:


#reading data dictionary


# In[17]:


Housing_dict = open('data_dict.txt')

print(Housing_dict.read())


# # DATA CLENSING

# In[19]:


#Checking for missing values and their treatment


# In[20]:


miss = round(housing.isna().sum()*100 / housing.shape[0], 2)
miss[miss>0].sort_values(ascending=False)


# In[ ]:


#all columns above have some missing values, top sex have highest missing values.
#Columns to be checked for how these missing values to be treated.


# In[22]:


miss_cols = miss[miss>0].sort_values(ascending=False).index


# In[23]:


miss_cols


# In[ ]:


#from data dictionary we know that NaN stands for not present for following features:
#'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'
#Imputing 'NP' for NaNs in these columns


# In[33]:


NP_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'MasVnrType', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']
housing[NP_cols]= housing[NP_cols].fillna('NP')


# In[34]:


housing.head()


# In[35]:


miss = round(housing.isna().sum()*100 / housing.shape[0], 2)
miss[miss>0].sort_values(ascending=False)


# In[36]:


#if Garage is not there, there will ne no GarageYrBlt


# In[37]:


housing[housing.GarageYrBlt.isna()]['GarageType'].value_counts(normalize= True)


# In[39]:


housing['GarageYrBlt']= housing['GarageYrBlt'].fillna(0)


# In[42]:


housing['GarageYrBlt']


# In[ ]:


#MSSubClass is a categorical data so changing its data type


# In[43]:


housing['MSSubClass']= housing['MSSubClass'].astype('object')


# In[47]:


housing.dtypes


# # EDA

# VIZUALIZING NUMERIC CATEGORIES

# In[49]:


sns.distplot(housing['SalePrice'])


# In[50]:


#Plotting numeric columns with Sale price


# In[52]:


numeric_cols= ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea',
'BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea']

sns.pairplot(housing, x_vars=['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})
sns.pairplot(housing, x_vars=['YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','LotFrontage'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})
sns.pairplot(housing, x_vars=['WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea'], y_vars='SalePrice', kind= 'reg', plot_kws={'line_kws':{'color':'teal'}})


# In[53]:


#Visualizing Categorical features


# In[55]:


cat_cols= ['OverallQual','GarageCars','ExterQual','BsmtQual','KitchenQual','FullBath','GarageFinish','FireplaceQu','Foundation','GarageType','Fireplaces','BsmtFinType1','HeatingQC']

plt.figure(figsize=[18, 40])

for i, col in enumerate(cat_cols, 1):
    plt.subplot(7,2,i)
    title_text= f'Box plot {col} vs SalePrice'
    x_label= f'{col}'
    fig= sns.boxplot(data= housing, x= col, y= 'SalePrice', palette= 'Greens')
    fig.set_title(title_text, fontdict= { 'fontsize': 18, 'color': 'Green'})
    fig.set_xlabel(x_label, fontdict= {'fontsize': 12, 'color': 'Brown'})
plt.show()


# In[60]:


plt.figure(figsize=[17,7])
sns.boxplot(data= housing, x= 'Neighborhood', y= 'SalePrice', palette= 'Greens')
plt.show()


# In[68]:


from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[71]:


types_train = housing.dtypes 
nums = types_train[(types_train == 'int64') | (types_train == float)]
cats = types_train[types_train == object]


# In[73]:


pd.DataFrame(types_train).reset_index().set_index(0).reset_index()[0].value_counts()


# In[74]:


#All numerical data in the dataset
num_list = list(nums.index)
print(num_list)


# In[75]:


#All catagorical data in the dataset
cat_list = list(cats.index)
print(cat_list)


# In[80]:


# Creating correlation heatmap
plt.figure(figsize = (20, 12))

sns.heatmap(housing.corr(numeric_only=True), annot= True, cmap= 'coolwarm', fmt= '.2f', vmin= -1, vmax= 1, )

plt.show()


# In[81]:


k = 10 
cols = (housing.corr(numeric_only=True)).nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(housing[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[82]:


#Infrences
#SalePrice is right sckewed and other numeic feature: 'GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea', 'BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','LotArea' have outlier and they all have somewhat linear relation with SalePrice.

#SalePrice is higher for the houses with higher OverallQual rating. Price reduces as quality decreases.

#SalePrice is high for houses having some specific Neighbourhoods : Northridge Heights, Stone Brook, Northridge etc.

#SalePrice is higher for the houses having Excellent Heating quality and median price reduces with Heating quality type and least for the houses having Poor heating quality.

#SalePrice is very high for Good Living Quarters type basement finished area and if the beasement height is more than 100+ inches and least for the houses not having basement.

#Houses having a garage as part of the house (typically has room above garage) and garage interior 'finish' or 'Rough Finished', have higest median SalePrice. Price is lower for the houses having no garage.

#Houses with garage in car capacity of 3 have highest SalePrice.

#Houses having Poured Contrete foundation has higher SalePrice. 

#SalePrice is high for houses with 3 Full bathrooms above grade.

#Corelation for GrLivArea and TotRmsAbvGrd= .83, GarageCars and GarageArea= .88 very high.


# In[83]:


#dropping GarageCars and TotRmsAbvGrd


# In[84]:


housing.drop(['GarageCars', 'TotRmsAbvGrd'], axis=1, inplace=True)
housing.shape


# # Data prep

# In[85]:


#treating target skewness

housing['SalePrice_log_trans']= np.log(housing['SalePrice'])


# In[87]:


housing.drop(['SalePrice','Id'], axis=1, inplace= True)
housing.shape


# In[89]:


sns.distplot(housing['SalePrice_log_trans'])
plt.show


# In[90]:


#train-test split

y= housing['SalePrice_log_trans']
X= housing.drop('SalePrice_log_trans', axis= 1)

X_train, X_test, y_train, y_test= train_test_split(X, y, train_size= .7, random_state= 42)


# In[91]:


train_index= X_train.index
test_index= X_test.index


# In[92]:


#Imputing rest of the features in test and train dataset using median (for continuous variables) 
#and mode (for categorical variables) calculated on train dataset.

housing['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace= True)
housing['LotFrontage'].fillna(X_train['LotFrontage'].median(), inplace= True)

housing['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace= True)
housing['MasVnrArea'].fillna(X_train['MasVnrArea'].median(), inplace= True)

housing['MasVnrType'].fillna(X_train['MasVnrType'].mode(), inplace= True)
housing['MasVnrType'].fillna(X_train['MasVnrType'].mode(), inplace= True)

housing['Electrical'].fillna(X_train['Electrical'].mode(), inplace= True)
housing['Electrical'].fillna(X_train['Electrical'].mode(), inplace= True)


# In[93]:


#Since the dataframe is changed, the encoding performed previously need to be redone on test train set
housing_cat= housing.select_dtypes(include= 'object')
housing_num= housing.select_dtypes(exclude= 'object')
housing_cat.describe()


# In[94]:


#encoding necessary unique data
housing['Street']= housing.Street.map(lambda x: 1 if x== 'Pave' else 0)
housing['Utilities']= housing.Utilities.map(lambda x: 1 if x== 'AllPub' else 0)
housing['CentralAir']= housing.CentralAir.map(lambda x: 1 if x== 'Y' else 0)

cat_cols= housing_cat.columns.tolist()
done_encoding= ['Street','Utilities', 'CentralAir']
cat_cols= [col for col in cat_cols if col not in done_encoding]
dummies= pd.get_dummies(housing[cat_cols], drop_first=True)


# In[95]:


housing.drop(cat_cols, axis=1, inplace= True)
housing= pd.concat([housing, dummies], axis= 1)


# In[96]:


#reconstructing train test data before scaling

X_train= housing.iloc[train_index, :].drop('SalePrice_log_trans', axis= 1)
y_train= housing.iloc[train_index, :]['SalePrice_log_trans']
X_test= housing.iloc[test_index, :].drop('SalePrice_log_trans', axis= 1)
y_test= housing.iloc[test_index, :]['SalePrice_log_trans']



# In[97]:


#Scaling

num_cols= housing_num.columns.tolist()
num_cols.remove('SalePrice_log_trans')
scaler= RobustScaler(quantile_range=(2, 98))
scaler.fit(X_train[num_cols])
X_train[num_cols]= scaler.transform(X_train[num_cols])
X_test[num_cols]= scaler.transform(X_test[num_cols])


# In[98]:


X_train[num_cols].head()


# # MODEL

# RIDGE

# In[99]:


#RANDOMLY CHOSEN ALPHAS

range1= [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
range2= list(range(2, 1001))
range1.extend(range2)
params_grid= {'alpha': range1}


# In[100]:


#Finding lambda 


# In[101]:


ridge= Ridge(random_state= 42)
gcv_ridge= GridSearchCV(estimator= ridge, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)      
gcv_ridge.fit(X_train, y_train)


# In[102]:


gcv_ridge.best_estimator_


# In[103]:


gcv_ridge.best_score_


# In[104]:


#fitting model for alpha=7


# In[105]:


ridge_model= gcv_ridge.best_estimator_
ridge_model.fit(X_train, y_train)


# In[106]:


#checking training set


# In[107]:


y_train_pred= ridge_model.predict(X_train)
print( 'r2 score on training dataset:', r2_score(y_train, y_train_pred))
print( 'MSE on training dataset:', mean_squared_error(y_train, y_train_pred))
print( 'RMSE on training dataset:', (mean_squared_error(y_train, y_train_pred)**.5))
print( 'MAE on training dataset:', mean_absolute_error(y_train, y_train_pred))


# In[108]:


ridge_model.coef_


# In[113]:


ridge_model.intercept_


# In[114]:


# Top 25 features with double the value of optimal alpha in Ridge
ridge_coef= pd.Series(ridge_model.coef_, index= X_train.columns)
top_25_ridge=  ridge_coef[abs(ridge_coef).nlargest(25).index]
top_25_ridge


# In[ ]:





# LASSO

# In[110]:


params_grid= {'alpha': range1}
lasso= Lasso(random_state= 42)
lasso_gcv= GridSearchCV(estimator= lasso, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)

lasso_gcv.fit(X_train, y_train)     


# In[111]:


lasso_gcv.best_estimator_


# In[112]:


lasso_gcv.best_score_


# In[115]:


#OPTIMAL ALPHA IS 0.0001

range3= [0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, .0002, .0003, .0004, .0005, .0006, .0007, .0008, .0009, .001]
params_grid= {'alpha': range3}
lasso_gcv= GridSearchCV(estimator= lasso, 
                        param_grid= params_grid,
                        cv= 3,
                        scoring= 'neg_mean_absolute_error',
                        return_train_score= True,
                        n_jobs= -1,
                        verbose= 1)

lasso_gcv.fit(X_train, y_train)   


# In[116]:


lasso_gcv.best_estimator_


# In[117]:


#OPTIMUM ALPHA IS 0.0004


# In[118]:


lasso_model= lasso_gcv.best_estimator_
lasso_model.fit(X_train, y_train)


# In[119]:


#EVALUATION ON TRAINING SET


# In[120]:


print( 'r2 score on training dataset:', r2_score(y_train, y_train_pred))
print( 'MSE on training dataset:', mean_squared_error(y_train, y_train_pred))
print( 'RMSE on training dataset:', (mean_squared_error(y_train, y_train_pred)**.5))
print( 'MAE on training dataset:', mean_absolute_error(y_train, y_train_pred))


# In[121]:


#EVALUATION ON TEST SET


# In[122]:


y_test_pred= lasso_model.predict(X_test)
print( 'r2 score on testing dataset:', r2_score(y_test, y_test_pred))
print( 'MSE on testing dataset:', mean_squared_error(y_test, y_test_pred))
print( 'RMSE on testing dataset:', (mean_squared_error(y_test, y_test_pred)**.5))
print( 'MAE on testing dataset:', mean_absolute_error(y_test, y_test_pred))


# In[123]:


#FEATURES USED IN LASSO and Ridge


# In[130]:


lasso_coef= pd.Series(lasso_model.coef_, index= X_train.columns)
selected_features= len(lasso_coef[lasso_coef != 0])
print('Features selected by Lasso:', selected_features)
print('Features present in Ridge:', X_train.shape[1])


# In[125]:


#FEATURES USED IN RIDGE


# In[126]:


print(X_train.shape[1])


# In[127]:


#INTERCEPT IN LASSO


# In[128]:


lasso_model.intercept_


# In[129]:


# Top 25 features with coefficients in Lasso model
top25_features_lasso=  lasso_coef[abs(lasso_coef[lasso_coef != 0]).nlargest(25).index]
top25_features_lasso


# In[ ]:


#Optimal alpha (lambda) value for Ridge Regression model is: 7
#Optimal alpha (lambda) value for Lasso Regression model is: 0.0004
#Ridge and Lasso have same accuracy so no overfitting
#Ridge use 270 and Lasso 134 features


# In[131]:


#Doubling the alphas


# In[133]:


ridge2= Ridge(alpha= 14, random_state= 42)
ridge2.fit(X_train, y_train)


# In[134]:


ridge_coef2= pd.Series(ridge2.coef_, index= X_train.columns)
top10_ridge2=  ridge_coef2[abs(ridge_coef2).nlargest(10).index]
top10_ridge2


# In[135]:


lasso2= Lasso(alpha= .0008, random_state=42)
lasso2.fit(X_train, y_train)


# In[136]:


lasso_coef2= pd.Series(lasso2.coef_, index= X_train.columns)
top10_lasso2=  lasso_coef2[abs(lasso_coef2[lasso_coef2 != 0]).nlargest(10).index]
top10_lasso2


# In[137]:


#top 5 features in lasso model


# In[138]:


top25_features_lasso.nlargest()


# In[139]:


#checking neighbourhood dummies and dropping them
cols_to_drop= X_train.columns[X_train.columns.str.startswith('Neighborhood')].tolist()
cols_to_drop.extend(['GrLivArea','OverallQual','OverallCond','GarageArea'])
cols_to_drop


# In[140]:


X_train= X_train.drop(cols_to_drop, axis= 1)
X_test= X_test.drop(cols_to_drop, axis= 1)
X_train.shape, X_test.shape


# In[141]:


#using lasso with these features


# In[142]:


lasso3= Lasso(alpha= .0004, random_state= 42)
lasso3.fit(X_train, y_train)


# In[143]:


lasso_coef3= pd.Series(lasso3.coef_, index= X_train.columns)
top5_lasso3=  lasso_coef3[abs(lasso_coef3[lasso_coef3 != 0]).nlargest().index]
top5_lasso3


# In[ ]:





# In[ ]:




