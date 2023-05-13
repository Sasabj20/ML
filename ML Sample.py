#!/usr/bin/env python


import pandas as pd
df1 = pd.read_html('https://en.wikipedia.org/wiki/List_of_largest_companies_of_South_Korea')
df = df1[0]

#Creating X - train and Y - test variables

import numpy as np
from sklearn.linear_model import LinearRegression




# Creating a new model and fitting it
print(df)
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# creating X - train and Y - test variables

print(df)

#X = df.iloc[[:,0,-1]]
X = df.iloc[:, [6]] #Employees
Y = df.iloc[:, [4]] #Revenue

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
#corr = df.corr()
heatmap = sns.heatmap(corr, annot=True, cmap="Blues")
print(reg.score(X,Y))
   Rank Fortune 500rank Name Industry \
0 1 19 Samsung Electronics Technology   
1 2 84 Hyundai Motor Automotive   
2 3 97 SK Holdings Holdings   
3 4 194 POSCO Steel   
4 5 207 LG Electronics Technology   
5 6 227 Korea Electric Power Corporation Utilities   
6 7 229 Kia Motors Automotive   
7 8 277 Hanwha Conglomerate   
8 9 385 Hyundai Mobis Automotive parts   
9 10 426 KB Financial Group Banking   
10 11 437 CJ Corporation Holdings   
11 12 447 GS Caltex Oil and Gas   
12 13 467 Samsung Life Insurance Insurance   
13 14 481 Samsung C&T Construction   

    Revenue(USD millions) Profits(USD millions) Employees Headquarters  
0 197705 18453 287439 Suwon  
1 90740 2557 114032 Seoul  
2 86163 616 108911 Seoul  
3 55592 1600 35261 Seoul  
4 53464 27 74000 Seoul  
5 50257 −2.013 47452 Naju  
6 49894 1567 52448 Seoul  
7 43258 77 57967 Seoul  
8 32649 1966 32065 Seoul  
9 29470 2842 26702 Seoul  
10 28986 229 59915 Seoul  
11 28541 388 3283 Seoul  
12 27291 839 5346 Seoul  
13 26396 901 16580 Seoul  
    Rank Fortune 500rank Name Industry \
0 1 19 Samsung Electronics Technology   
1 2 84 Hyundai Motor Automotive   
2 3 97 SK Holdings Holdings   
3 4 194 POSCO Steel   
4 5 207 LG Electronics Technology   
5 6 227 Korea Electric Power Corporation Utilities   
6 7 229 Kia Motors Automotive   
7 8 277 Hanwha Conglomerate   
8 9 385 Hyundai Mobis Automotive parts   
9 10 426 KB Financial Group Banking   
10 11 437 CJ Corporation Holdings   
11 12 447 GS Caltex Oil and Gas   
12 13 467 Samsung Life Insurance Insurance   
13 14 481 Samsung C&T Construction   

    Revenue(USD millions) Profits(USD millions) Employees Headquarters  
0 197705 18453 287439 Suwon  
1 90740 2557 114032 Seoul  
2 86163 616 108911 Seoul  
3 55592 1600 35261 Seoul  
4 53464 27 74000 Seoul  
5 50257 −2.013 47452 Naju  
6 49894 1567 52448 Seoul  
7 43258 77 57967 Seoul  
8 32649 1966 32065 Seoul  
9 29470 2842 26702 Seoul  
10 28986 229 59915 Seoul  
11 28541 388 3283 Seoul  
12 27291 839 5346 Seoul  
13 26396 901 16580 Seoul  
    Employees
0 287439
1 114032
2 108911
3 35261
4 74000
5 47452
6 52448
7 57967
8 32065
9 26702
10 59915
11 3283
12 5346
13 16580
    Revenue (USD millions)
0 197705
1 90740
2 86163
3 55592
4 53464
5 50257
6 49894
7 43258
8 32649
9 29470
10 28986
11 28541
12 27291
13 26396
[[0.61497113]]
[16697.92777392]
0.9517524935850562











