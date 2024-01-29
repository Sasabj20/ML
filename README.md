# ML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#import dataset
Table=pd.read_excel('E:Table.xlsx')
Table.describe()
print(Table.head())

x = Table[['Administration','Marketing Spend']]
y = Table['Profit']

#from sklearn.compose import make_column_transformer
#col_trans=make_column_transformer(
   # (OneHotEncoder(handle_unknown='ignore'),['State']),
#    remainder='passthrough')
#x=col_trans.fit_transform(x)

print(x)
print(y)
print(Table[['State', 'Profit']])
print(Table)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)

linreg=LinearRegression()
linreg.fit(x_train,y_train)
#Predict the Test Results
#Use the predict method to predict the results, then pass the independent variables into it and view the results. It will give the array with all the values in it.
print(linreg.coef_)
print(linreg.intercept_)
y_pred=linreg.predict(x_test)
y_pred

Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)  



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
​
#import dataset
Table=pd.read_excel('E:Table.xlsx')
Table.describe()
print(Table.head())
​
x = Table[['Administration','Marketing Spend']]
y = Table['Profit']
​
#from sklearn.compose import make_column_transformer
#col_trans=make_column_transformer(
   # (OneHotEncoder(handle_unknown='ignore'),['State']),
#    remainder='passthrough')
#x=col_trans.fit_transform(x)
​
print(x)
print(y)
print(Table[['State', 'Profit']])
print(Table)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)
​
linreg=LinearRegression()
linreg.fit(x_train,y_train)
#Predict the Test Results
#Use the predict method to predict the results, then pass the independent variables into it and view the results. It will give the array with all the values in it.
print(linreg.coef_)
print(linreg.intercept_)
y_pred=linreg.predict(x_test)
y_pred
​
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)
       State  R&D Spend  Administration  Marketing Spend  Profit
0       Ohio        500              19             1100     450
1   Michigen        600              22             1900     617
2      Idaho        800              25             2000     647
3      Idaho       1400              29             3200     860
4  Michigen         660              13             1300     460
    Administration  Marketing Spend
0               19             1100
1               22             1900
2               25             2000
3               29             3200
4               13             1300
5               18             1700
6               29             2300
7               38             3600
8               35             3400
9               14             1450
10              16             1600
11              26             2200
12              38             2600
13              41             2800
14              15             1520
15              17             2000
16              19             2200
0     450
1     617
2     647
3     860
4     460
5     545
6     742
7     895
8     880
9     480
10    520
11    722
12    800
13    820
14    500
15    650
16    800
Name: Profit, dtype: int64
        State  Profit
0        Ohio     450
1    Michigen     617
2       Idaho     647
3       Idaho     860
4   Michigen      460
5        Ohio     545
6       Idaho     742
7   Michigen      895
8        Ohio     880
9       Idaho     480
10  Michigen      520
11       Ohio     722
12       Ohio     800
13   Michigen     820
14      Idaho     500
15   Michigen     650
16       Ohio     800
        State  R&D Spend  Administration  Marketing Spend  Profit
0        Ohio        500              19             1100     450
1    Michigen        600              22             1900     617
2       Idaho        800              25             2000     647
3       Idaho       1400              29             3200     860
4   Michigen         660              13             1300     460
5        Ohio        800              18             1700     545
6       Idaho       1000              29             2300     742
7   Michigen        1600              38             3600     895
8        Ohio       1500              35             3400     880
9       Idaho        580              14             1450     480
10  Michigen         600              16             1600     520
11       Ohio        920              26             2200     722
12       Ohio       1100              38             2600     800
13   Michigen       1300              41             2800     820
14      Idaho        500              15             1520     500
15   Michigen        550              17             2000     650
16       Ohio        620              19             2200     800
X_train: (13, 2)
X_test: (4, 2)
Y_train: (13,)
Y_test: (4,)
[1.1349916  0.19250487]
227.47131065269787
 Accuracy of the model is 94.24
