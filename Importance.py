# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:26:27 2020

@author: Yuki
"""


import  xgboost as xgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test, y_train,y_test = train_test_split(x, y,test_size = 1/5.,random_state = 8)

xgb_train = xgb.DMatrix(x_train,label = y_train)
xgb_test = xgb.DMatrix(x_test, label = y_test)

params ={
    "objective":"binary:logistic",
    "booster":"gbtree",
    "eta":0.1,
    "max_depth":5
    }
num_round = 50

watchlist=[(xgb_test,'eval'),(xgb_train,'train')]

bst = xgb.train(params, xgb_train, num_round, watchlist)

importance = bst.get_fscore()

importance = sorted(importance.items(), key = lambda x: x[1],reverse = True)

df = pd.DataFrame(importance,columns = ['feature','fscore'])
print(df)
