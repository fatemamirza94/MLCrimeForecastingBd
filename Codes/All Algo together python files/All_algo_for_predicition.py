# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:56:46 2019

@author: Shibli
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error 


data = {'2010':[38,	37,	207,	112,	3,	991,	65,	13,	214,	374,	6561,	33,	5,	334,	33],
        '2011':[33,	43,	194,	473,	8,	1060,	60,	20,	214,	331,	6611,	41,	6,	350,	19],
        '2012':[34,	61,	170,	232,	1,	1168,	55,	13,	186,	277,	6879,	44,	11,	444,	56],
        '2013':[40,	31,	209,	115,	1,	1185,	62,	44,	154,	244,	6920,	27,	15,	502,	28],
        '2014':[36,	37,	200,	51,	0,	1478,	60,	8,	163,	225,	6892,	36,	19,	813,	31],
        '2015':[31,	47,	175,	58,	1,	1497,	53,	20,	156,	266,	6767,	26,	22,	1019,	27],
        '2016':[29,	36,	166,	52,	1,	1538,	38,	22,	155,	187,	6130,	58,	12,	1679,	12],
        '2017':[22,	36,	169,	62,	0,	1451,	20,	16,	161,	182,	5763,	59,	8,	3441,	5]}

# Create DataFrame
df = pd.DataFrame(data)
#print(df)

target = {'2018':[20,	24,	166,	55,	0,	1321,	20,	20,	132,	205,	4975,	54,	56,	4384,	2]}

################# Linear Reg Begin ############
X = df
y = target['2018']

lm = LinearRegression()
modellm = lm.fit(X,y)

predictionslm = lm.predict(X)
print('=================================================================')
print('Predicted values for target')
print(predictionslm)

accuracylm = lm.score(X,y)
print('linear regression score')
print(accuracylm*100,'%')
########### Linear Regression End ######################

########### Logistic Regression Begins ######################
logm = LogisticRegression()
modellogm = logm.fit(X,y)

print('=================================================================')
print('Predicted values for target')
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
print('=================================================================')
print('Predicted values for target')
print(predictionsam)

accuracyam = am.score(X,y)
print('adaboost regression score')
print(accuracyam*100,'%')
########## Adaboost End ###############################

########## gradient boosting Begin #####################
gm = GradientBoostingRegressor()
modelgm = gm.fit(X,y)

predictionsgm = gm.predict(X)
print('=================================================================')
print('Predicted values for target')
print(predictionsgm)

accuracygm = gm.score(X,y)
print('grad boost regression score')
print(accuracygm*100,'%')

########## gradient boosting end #####################

########## xgboosting Begin #####################
xgm = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.7,
                max_depth = 3, alpha = 10, n_estimators = 15)
modelxgm = xgm.fit(X,y)

predictionsxgm = xgm.predict(X)
print('=================================================================')
print('Predicted values for target')
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
print('=================================================================')
print('Predicted values for target')
print(predictionsrm)

accuracyrm = rm.score(X,y)
print('regression tree score')
print(accuracyrm*100,'%')

########## regression tree end #####################

#########Support Vector Regressor Begin################
from sklearn.svm import SVR
regressor=SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.5,
               coef0=1)

regressor.fit(X,y)

predictionSVR = regressor.predict(X);
print('=================================================================')
print('Predicted values for target')
print(predictionSVR)

accuracySVR = regressor.score(X,y)
print('Support Vector Regression Score')
print(accuracySVR*100,'%')

########Support Vector Regressor End###############

##########Random Forest Regressor Begin###########
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 15, criterion='mse', max_depth=3, random_state=1)

RF.fit(X,y)

predictionRF = RF.predict(X)
print('=================================================================')
print('Predicted values for target')
print(predictionRF)

accuracyRF = RF.score(X, y)
print('Random Forest Regressor Score')
print(accuracyRF*100, '%')

##########Random Forest Regressor End###########

#########Multi Layer Perceptron Regression Begin##########
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation='tanh', max_iter=200, solver='adam', learning_rate='adaptive', alpha=1e-5,hidden_layer_sizes=(30, 30, 30 ), random_state=42)

mlp.fit(X, y)
y_pred_naive = mlp.predict(X)
print('=================================================================')
print('Predicted values for target')
print(y_pred_naive)

accuracy_mlp = mlp.score(X, y)
print('Multi Layer Perceptron Score')
print(accuracy_mlp*100, '%')
print('=================================================================')
###########Multi Layer Perceptron Regression End########