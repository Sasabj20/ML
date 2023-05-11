#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import statsmodels.formula.api as sm

df = pd.read_excel('E:Tabela29II.xlsx')
rege = sm.ols(formula="Salary ~ Bonuses + Contribution", data=df).fit()
print(df)
X = df[["Bonuses","Contribution"]]
y = df["Salary"]
from sklearn.linear_model import LinearRegression
# Note the difference in argument order
print(X)
print(y)

#predictions = model.predict(X) # make the predictions by the model

# Print out the statistics

rege.fit = (X, y)
rege = LinearRegression().fit(X, y)
lr = LinearRegression()
#x, y = data[:, :-1], data[:, -1]
lr.fit(X, y)
lr = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
print(lr.coef_)

#print (model.coef_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

model = LinearRegression().fit(X_train, y_train)
model.fit = (X_train, y_train)
print(model.coef_)







