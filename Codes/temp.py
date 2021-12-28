

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
 
# intialise data of lists.
"""
data = pd.read_csv('CrimeData.csv')  # load data set
X = data.iloc[:,0].values.reshape(-1, 1)  # values converts it into a numpy array
target = data.iloc[:,1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
"""

data = {'2010':[47,220,245,363,3,1370,139,155,555,1915,7228,518,82,10535,144],
        '2011':[43,294,259,324,9,1538,118,142,589,2050,6766,358,77,10405,88],
        '2012':[66,222,264,388,11,1637,149,136,592,2240,8296,284,94,8345,99]}
 
# Create DataFrame
df = pd.DataFrame(data)

target = {'2013':[47,241,270,338,37,1631,165,203,534,2196,7077,242,387,7927,106]}
 
# Print the output.
#print(df)

xf = pd.DataFrame(target)

print(xf)
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target)


#X = df
#y = target['2013']

#lm = LinearRegression()
#model = lm.fit(X,y)

#predictions = lm.predict(X)
#print(predictions)


#print(lm.coef_)
#print(lm.intercept_)

#accuracy = lm.score(X,y)
#print(accuracy*100,'%')


"""