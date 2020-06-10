# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:26:04 2020

@author: Yuki
"""

#mashroom classification
def loadfmap(fname):
    fmap = {}
    nmap = {}

    #以空字符位分隔符分隔一行
    for l in open(fname):
        arr=l.split()
        #解析每行中的特征名称，取值
        #idx为初始特征索引，ftype为初始特征名称，content为该特征值说明
        if arr[0].find('.') != -1:
            idx = int( arr[0].strip('.'))
            assert idx not in fmap
            fmap[idx] = {}
            ftype = arr[1].strip(':')
            content = arr[2]
        else:
            content = arr[0]
            #解析取值说明
            #fmap视为特征的每个取值分配一个唯一标示的索引，nmap为处理后新特征重新命名
        for it in content.split(','):
            if it.strip() == '':
                continue
            k,v = it.split('=')
            fmap[idx][v] = len(nmap) + 1
            nmap[len(nmap)] = ftype + '=' + k
    return fmap, nmap

def write_nmap(fo,nmap):
    for i in range(len(nmap)):
        fo.write('%d\t%s\ti\n' % (i,nmap[i]))
        #解析特征描述文件
fmap,nmap = loadfmap('C:\\xgb\\xgboost-master\\demo\\binary_classification\\agaricus-lepiota.fmap')
        #保存处理后的新特征索引和名称的映射
fo = open('featmap.txt','w')
for l in open('C:\\xgb\\xgboost-master\\demo\\binary_classification\\agaricus-lepiota.data'):
    arr = l.split(',')
    if arr[0] == 'p':
        fo.write('1')
    else:
        assert arr[0] =='e'
        fo.write('0')
    for i in range(1,len(arr)):
        fo.write('%d:1' % fmap[i][arr[i].strip()])
    fo.write('\n')
    
fo.close()

import xgboost as xgb

xgb_train = xgb.DMatrix("C:\\xgb\\xgboost-master\\demo\\data\\agaricus.txt.train")
xgb_test = xgb.DMatrix("C:\\xgb\\xgboost-master\\demo\\data\\agaricus.txt.test")

params = {
    "objective": "binary:logistic",
    "booster":"gbtree",
    "eta":1.0,
    "gamma":1.0,
    "min_child_weight":1,
    "max_depth":3
    }

num_round = 2
watchlist = [(xgb_train,'train'),(xgb_test,'test')]
model = xgb.train(params, xgb_train, num_round, watchlist)
model.save_model("./0002.model")

#Load model to predict

bst = xgb.Booster()
bst.load_model("./0002.model")
pred = bst.predict(xgb_test)
print(pred)

dump_model = bst.dump_model("./dump.raw.txt")
dump_model1 = bst.dump_model("./dump.nice.txt","C:\\xgb\\xgboost-master\\demo\\data\\featmap.txt")
pip install wget
import wget
#多分类问题
wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt") 
import pandas as pd
import xgboost as xgb
import numpy as np

data = pd.read_csv('./seeds_dataset.txt',header=None,sep='\s+',converters={7:lambda x:int(x)-1})
data.rename(columns = {7:'label'},inplace=True)
data.head(10)
mask = np.random.rand(len(data)) < 0.8
train = data[mask]
test = data[~mask]

xgb_train = xgb.DMatrix(train.iloc[:,:6], label = train.label)
xgb_test = xgb.DMatrix(test.iloc[:,:6], label = test.label)

params = {
    "objective": "multi:softmax",
    "eta":0.1,
    "num_class":3,
    "max_depth":5
    }
watchlist = [(xgb_train,'train'),(xgb_test,'test')]
num_round = 50
bst = xgb.train(params,xgb_train, num_round,watchlist)

pred = bst.predict(xgb_test)
error_rate = np.sum(pred != test.label) / test.shape[0]
print('softmax error rate: {}'.format(error_rate))


#softprob
params = {
    "objective": "multi:softprob",
    "eta":0.1,
    "num_class":3,
    "max_depth":5
    }
bst = xgb.train(params,xgb_train, num_round,watchlist)
pred_prob = bst.predict(xgb_test)
print(pred_prob)
pred_label = np.argmax(pred_prob,axis = 1)
print(pred_label)
error_rate = np.sum(pred_label != test.label )/ test.shape[0]
print(error_rate)
