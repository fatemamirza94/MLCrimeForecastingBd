# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:02:40 2019

@author: GL62M
"""

import pandas as pd
from sklearn import datasets

boston = datasets.load_boston() 

import pandas as pd

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

X, y = data.iloc[:,:-1],data.iloc[:,-1]

data_dmatrix = xgb.DMatrix(data=X,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

expected = y_test

import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, preds)
plt.plot([0, 20], [0, 20], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))