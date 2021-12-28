# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:11:01 2019

@author: GL62M
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error 

data = {'2010':[47,220,245,363,3,1370,139,155,555,1915,7228,518,82,10535,144],
        '2011':[43,294,259,324,9,1538,118,142,589,2050,6766,358,77,10405,88],
        '2012':[66,222,264,388,11,1637,149,136,592,2240,8296,284,94,8345,99],
        '2013':[47,241,270,338,37,1631,165,203,534,2196,7077,242,387,7927,106],
        '2014':[47,265,262,332,7,1611,176,104	,650,2130,6219,225,104,7063,222],
        '2015':[45,205,239,226,26,1550,146,118,642,1711,5795,263,195,8365,	201	],
        '2016':[23,131,165,217,1,1651,103,72,547	,1516,5407,244,88	,9627,108],
        '2017':[20,103,218,202,7,1779,85,132,554	,1197,5315,157,108,13638,115]}

# Create DataFrame
df = pd.DataFrame(data)
#print(df)

target = {'2018':[17,83,216,216	,17,	1782,75,236,613,1290	,5708,155,354,16215,173]}

################# Linear Reg Begin ############
lm = LinearRegression()
modellm = lm.fit(X,y)

predictionslm = lm.predict(X)
print(predictionslm)

#print(lm.coef_)
#print(lm.intercept_)

accuracylm = lm.score(X,y)
print('linear regression score')
print(accuracylm*100,'%')

########### Linear Regression End ######################

########### Logistic Regression Begins ######################
logm = LogisticRegression()
modellogm = logm.fit(X,y)

predictionslogm = logm.predict(X)
print(predictionslogm)

#print(lm.coef_)
#print(lm.intercept_)

accuracylogm = logm.score(X,y)
print('logistic regression score')
print(accuracylogm*100,'%')

########## Adaboost Begin ###############################
am = AdaBoostRegressor()
modelam = am.fit(X,y)

predictionsam = am.predict(X)
print(predictionsam)

accuracyam = am.score(X,y)
print('adaboost regression score')
print(accuracyam*100,'%')
########## Adaboost End ###############################

########## gradient boosting Begin #####################
gm = GradientBoostingRegressor()
modelgm = gm.fit(X,y)

predictionsgm = gm.predict(X)
print(predictionsgm)

accuracygm = gm.score(X,y)
print('grad boost regression score')
print(accuracygm*100,'%')

########## gradient boosting end #####################

########## xgboosting Begin #####################
xgm = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
modelxgm = xgm.fit(X,y)

predictionsxgm = xgm.predict(X)
print(predictionsxgm)

accuracyxgm = xgm.score(X,y)
print('x grad boost regression score')
print(accuracyxgm*100,'%')
########## xgboosting end #####################


########## regression tree Begin #####################
rm = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor
                             max_depth=3)  
modelrm = rm.fit(X,y)

predictionsrm = rm.predict(X)
print(predictionsrm)

accuracyrm = rm.score(X,y)
print('regression tree score')
print(accuracyrm*100,'%')

########## regression tree end #####################

