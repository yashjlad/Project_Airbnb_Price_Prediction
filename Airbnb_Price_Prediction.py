#!/usr/bin/env python
# coding: utf-8

# In[132]:


import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random


# In[20]:


data = pd.read_csv(r"C:\Users\yash_\OneDrive\Desktop\Projects\House price prediction\s3_files\frankfurt\tomslee_airbnb_frankfurt_1360_2017-06-22.csv") 
pd.set_option('display.max_columns', None)


# In[24]:


data.head(7)


# In[22]:


data.describe(include='all')


# In[23]:


data.describe(include='all')


# In[34]:


data = data.drop(['country','borough','bathrooms','minstay','last_modified','location'], axis=1)


# In[35]:


data.head()


# In[36]:


data = data.drop(['city','room_id','survey_id','host_id'], axis=1)


# In[37]:


data


# In[40]:


data.isnull().sum()


# ## Data Preparation 

# In[ ]:





# ## Explanatory Data Analysis

# ### Inspecting Price range of the houses

# In[55]:


plt.figure(figsize=(18,10))
sb.distplot(data['price'])


# ### Comparing price by the type of room

# #### Types of room available

# In[59]:


data['room_type'].unique()


# #### Price according to the type of room

# In[63]:


data.boxplot(column='price', by='room_type', figsize=(20,10))


# In[64]:


data.boxplot(column='price', by='room_type', figsize=(20,10), rot=90)


# In[65]:


data.boxplot(column='price', by='neighborhood', figsize=(20,10), rot=90)


# ### Scatterplot of the prices in different locations according to the latitude and longtitude

# In[72]:


data.plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.5, figsize=(10,10))


# In[95]:


data[data['price']< 200].plot.scatter(x='longitude', y='latitude', c='price', cmap='cool', alpha=0.8, figsize=(10,10))


# ### Reviews 

# In[97]:


frankfurt.plot.scatter(x='reviews', y='price', figsize=(10,8));


# In[106]:


plt.scatter(np.log(data['reviews']+1), data['price'] )
plt.figure(figsize=(20,8))


# ### Satisfaction

# In[109]:


frankfurt.plot.scatter(x='overall_satisfaction', y='price', figsize=(10,8));


# ### Accomodates

# In[111]:


frankfurt.plot.scatter(x='accommodates', y='price', figsize=(10,8));


# ## Feature Engineering 

# ### Adding 2 columns: logreviews, bedrooms_per_person

# In[113]:


data['logreviews'] = np.log(1+data['reviews'])
data['bedrooms_per_person']= data['bedrooms']/data['accommodates']


# ### Removing properties without reviews 

# In[117]:


frankfurt=data.copy()
###Remove the properties without any reviews 


# In[116]:


data= pd.get_dummies(data)
data


# In[118]:


X= data.copy().drop('price', axis=1)


# In[119]:


Y= data['price'].copy()


# In[120]:


X


# In[121]:


Y


# ## Scaling the dataset

# In[122]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# In[123]:


scaler= StandardScaler()
scaler.fit(X_train)
X_train_scaled= scaler.transform(X_train)
X_test_scaled= scaler.transform(X_test)


# ## Baseline

# In[131]:


baseline = Y_train.median() #median train
print('If we just take the median value, our baseline, we would say that an overnight stay in Frankfurt costs: ' + str(baseline))


# In[135]:


baseline_error = np.sqrt(mean_squared_error(y_pred=np.ones_like(Y_test) * baseline, y_true=Y_test))


# In[137]:


baseline_error
print('And we will be right +- ' + str(baseline_error))

