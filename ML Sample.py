#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
df1 = pd.read_html('https://en.wikipedia.org/wiki/List_of_largest_companies_of_South_Korea')
df = df1[0]

#Creating X - train and Y - test variables

import numpy as np
from sklearn.linear_model import LinearRegression


# Splitting the data into training and testing

# Creating a new model and fitting it
print(df)
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# creating X - train and Y - test variables

print(df)

#X = df.iloc[[:,0,-1]]
X = df.iloc[:, [6]]
Y = df.iloc[:, [4]]

print(X)
print(Y)
from sklearn.linear_model import LinearRegression

# 2. instantiate model
reg = LinearRegression().fit(X,Y)

# 3. fit 
reg.fit = (X,Y)
print(reg.coef_)
print(reg.intercept_)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso




plt.figure(figsize=(14,8))
corr = df.corr()
heatmap = sns.heatmap(corr, annot=True, cmap="Blues")


# In[ ]:





# In[ ]:




