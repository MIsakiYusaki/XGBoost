# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:55:04 2020

@author: Yuki
"""

import wget
wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls")  

import pandas as pd                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
import xgboost as xgb
import numpy as np

data = pd.read_excel('./Concrete_Data.xls')
data.head(10)
#data.rename(columns={'Concrete compressive strength(MPa, megapascals)':'label'},inplace = True)
data.head()
mask = np.random.rand(len(data))<0.8
train = data[mask]
test = data[~mask]
train.head()
xgb_train = xgb.DMatrix(train.iloc[:,:7],label=train.label)
xgb_test = xgb.DMatrix(test.iloc[:,:7],label=test.label)

params = {
    "objective":"reg:linear",
    "booster":"gbtree",
    "eta":0.1,
    "min_child_weight":1,
    "max_depth":5
        }
num_round = 50
watchlist = [(xgb_train,'train'),(xgb_test,'test')]
model = xgb.train(params,xgb_train,num_round,watchlist)

model.save_model("./model.xgb")
bst = xgb.Booster()
bst.load_model("./model.xgb")
pred = bst.predict(xgb_test)
print(pred)

dump_model = bst.dump_model("./dump.txt")
