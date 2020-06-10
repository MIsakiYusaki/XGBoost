# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:48:27 2020

@author: Yuki
"""


import xgboost as xgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/5.,random_state = 8)

xgb_train = xgb.DMatrix(x_train,label = y_train)
xgb_test = xgb.DMatrix(x_test,label = y_test)

params = {
    "booster":"gbtree",
    "eta":0.1,
    "max_depth":5
    }
num_round = 50

def logregobj(preds,dtrain):
    labels = dtrain.get_label()
    preds = 1.0/(1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess

def evalerror(preds,dtrain):
    labels = dtrain.get_label()
    return 'error',float(sum(labels != (preds > 0.0))) / len(labels)
watchlist = [(xgb_train,'train'),(xgb_test,'test')]

bst = xgb.train(params,xgb_train,num_round,watchlist,obj = logregobj,feval = evalerror)

res = xgb.cv(params,xgb_train,num_round,nfold=5,metrics={'auc'},seed = 0,callbacks = [xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(5)])


#自定义预处理函数
def fpreproc(xgb_train,xgb_test,params):
    label = xgb_train.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    params['scale_pos_weight'] = ratio
    return (xgb_train,xgb_test,params)

xgb.cv(params,xgb_train,num_round,nfold = 5, metrics={'auc'},seed = 0,fpreproc = fpreproc,callbacks = [xgb.callback.early_stop(5)])
