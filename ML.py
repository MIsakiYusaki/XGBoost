# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:52:03 2020

@author: Yuki
"""
import numpy as np
import pandas as pd

iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',sep = ",", names = ['sepal_length','sepal_width','petal_length','petal_width','class'])

iris_data.head(10)

from matplotlib import pyplot as plt

grouped_data = iris_data.groupby("class")

group_mean =grouped_data.mean()

group_mean.plot(kind="bar")
plt.legend(loc="center right",bbox_to_anchor=(1.4,0.3),ncol = 1)
plt.show()

msk = np.random.rand(len(iris_data)) < 0.8

train_data_origin = iris_data[msk]
test_data_origin = iris_data[~msk]
train_data = train_data_origin.reset_index(drop=True)
test_data = test_data_origin.reset_index(drop=True)
train_label = train_data['class']
test_label = test_data['class']

train_fea = train_data.drop('class',1)
test_fea = test_data.drop('class',1)

train_norm = (train_fea - train_fea.min()) / (train_fea.max() - train_fea.min())
test_norm = (test_fea - test_fea.min()) / (test_fea.max() - test_fea.min())

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct +=1
            return (1-(correct/float(len(testSet)))) * 100
        
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_norm, train_label)
predict = knn.predict(test_norm)
accuracy = getAccuracy(test_label, predict)
print("Accuracy:"+repr(accuracy) + "%")


from sklearn import datasets
boston = datasets.load_boston()
x = boston.data
y = boston.target



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/5.,random_state = 8)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test,y_pred)

print("MSE:" + repr(mse))

import pandas as pd
import xgboost as xgb
import numpy as np
train_xgb = xgb.DMatrix(x_train,y_train)
params = {"objective":"reg:linear","booster":"gbtree"}
model = xgb.train(dtrain = train_xgb,params=params)
y_pred = model.predict(xgb.DMatrix(x_test))
y_pred
mse = mean_squared_error(y_test,y_pred)
mse




# Logistic Regression Cancer prediction
#sklearn
from sklearn import datasets
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
dir(cancer)
y
#x1 = pd.DataFrame(x)
#x1.isnull().any().sum()
y = y[0:506]
len(x)
len(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/5.,random_state = 8)

y_train
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
from sklearn.metrics import classification_report
print (classification_report(y_test,y_pred,target_names=['Benign','Malignant']))

#XGboost
import pandas as pd
import xgboost as xgb
import numpy as np

train_xgb = xgb.DMatrix(x_train,y_train)
params = {"objective":"reg:logistic","booster":"gblinear"}
model = xgb.train(dtrain = train_xgb,params=params)
y_pred = model.predict(xgb.DMatrix(x_test))

ypred_bst = np.array(y_pred)
ypred_bst = ypred_bst >0.5
ypred_bst
ypred_bst = ypred_bst.astype(int)
ypred_bst

#decesion tree

from sklearn import datasets
cancer = datasets.load_breast_cancer()
x = cancer.data

y = cancer.target
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/5.,random_state = 8)
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

clf = tree.DecisionTreeClassifier(max_depth = 4)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test,y_pred,target_names=["Benign",'Malignant']))


import graphviz
from sklearn.tree import DecisionTreeClassifier
dot_data = tree.export_graphviz(clf,out_file="tree.dot",feature_names = cancer.feature_names,class_names=cancer.target_names,
                                filled = True,rounded = True,special_characters = True)
dot_data
graph = graphviz.Source(dot_data)
print(graph)

