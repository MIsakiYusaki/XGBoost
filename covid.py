# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:41:33 2020

@author: Yuki
"""
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.simplefilter('ignore')

#载入文件
train = pd.read_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train.csv') 
test = pd.read_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\test.csv')
population = pd.read_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\datasets_507962_1091873_population_by_country_2020.csv')

pd.options.display.max_columns = 30
#处理缺失值
train['Province_State'].fillna('None',inplace=True)
train['County'].fillna("None",inplace=True)
train.drop(['Id','Weight'],inplace=True, axis=1)
train.head(10)
#分离确诊和死亡
train_Confirmed = train[train['Target']=='ConfirmedCases']
train_Fatalities = train[train['Target']=='Fatalities']
#合并数据框前前预处理
train_Confirmed.rename(columns={'TargetValue':'ConfirmedCases'},inplace = True)
train_Fatalities.rename(columns={'TargetValue':'Fatalities'},inplace = True)
train_Confirmed.drop('Target',axis=1,inplace = True)
train_Confirmed
train_Fatalities.drop('Target',axis=1,inplace = True)
train_Fatalities



#合并数据框
train_modifided = pd.merge(train_Confirmed,train_Fatalities,on = ['County','Province_State','Country_Region','Population','Date'])
#train_modifided.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train_modifided.csv')

#不要以下四个地方的数据

dropvalues = ('Diamond Princess', 'Kosovo', 'MS Zaandam', 'West Bank and Gaza')
b1 = train_modifided['Country_Region'] == dropvalues ##产生布尔值

for v in dropvalues:
    train_modifided = train_modifided.drop(index = (train_modifided.loc[(train_modifided['Country_Region']==v)].index))




train_modifided

#train_Confirmed.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\Confirmed_train.csv')
#train_Fatalities.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\Fatalities_train.csv')
u1 = train_modifided['Country_Region'].unique()
u2 = population['Country (or dependency)'].unique()
s1 = set(u1)
s2 = set(u2)
s1.difference(s2)

#丰富数据

train_enriched = pd.merge(train_modifided,population,left_on=['Country_Region'],right_on=['Country (or dependency)'],how = 'left')

train_enriched[351664:351670]
train_enriched.drop(['Country (or dependency)','Population'],inplace = True,axis = 1)

train_enriched.columns
#重排
order = ['Date','Country_Region','Province_State','County','Population (2020)','Yearly Change','Net Change',
         'Density (P/Km²)','Land Area (Km²)','Migrants (net)','Fert. Rate','Med. Age','Urban Pop %','World Share','ConfirmedCases','Fatalities']
train_enriched = train_enriched[order]
#输出丰富后的数据


#将分类型变量改为int

train_enriched['Country_Region'] = train_enriched['Country_Region'].astype('category').values.codes
train_enriched['Province_State'] = train_enriched['Province_State'].astype('category').values.codes
train_enriched['Date'] = train_enriched['Date'].astype('category').values.codes
train_enriched['County'] = train_enriched['County'].astype('category').values.codes

Index = ['Date', 'Country_Region', 'Province_State', 'County',
       'Population (2020)', 'Yearly Change', 'Net Change', 'Density (P/Km²)',
       'Land Area (Km²)', 'Migrants (net)', 'Fert. Rate', 'Med. Age',
       'Urban Pop %', 'World Share', 'ConfirmedCases', 'Fatalities']




Convert = ['Yearly Change', 'Fert. Rate', 'Med. Age', 'Urban Pop %','World Share']



for v in Convert:
    train_enriched[v] = train_enriched[v].str.strip('%')

train_enriched = train_enriched.apply(pd.to_numeric, errors='coerce')

for v in Convert:
    train_enriched[v] = train_enriched[v].astype(np.float64)




train_enriched[Convert].isna()

train_enriched.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train_enriched.csv')
##数据预处理完成

#尝试使用XGBOOST进行回归
import xgboost as xgb
from sklearn.model_selection import train_test_split

variables = ['Date','Country_Region','Province_State','County','Population (2020)','Yearly Change','Net Change',
         'Density (P/Km²)','Land Area (Km²)','Migrants (net)','Fert. Rate','Med. Age','Urban Pop %','World Share']

Target1 = ['ConfirmedCases']
Target2 = ['Fatalities']

x = train_enriched[variables]
y1 = train_enriched[Target1]
y2 = train_enriched[Target2]


x
y1
y2

x_train,x_test,y1_train,y1_test = train_test_split(x,y1,test_size = 1/5.,random_state = 8)
x_train
x_test
y1_train
y1_test
xgb_train = xgb.DMatrix(x_train,label = y1_train)
xgb_test = xgb.DMatrix(x_test,label = y1_test)
xgb_train
xgb_test
params = {
    "objective":"reg:linear",
    "booster":"gbtree",
    "eta":0.1,
    "min_child_weight":1,
    "max_depth":5
    }


num_round = 50

watchlist = [(xgb_train,'train'),(xgb_test,'test')]
model = xgb.train(params, xgb_train, xgb_test, num_round)
 
    
    def model_cv(bst,train,features,nfold=5,early_stopping_rounds=30):
    params = bst.get_xgb_params()
    train = xgb.DMatrix(train[features].values,train[label].values)
    
    #交叉验证
    cv_result = xgb.cv(params,
                       train,
                       num_boost_round = bst.get_xgb_params()['n_estimators'],
                       nfold = nfold,
                       metrics = ['rmse'],
                       early_stopping_rounds = early_stopping_rounds)
    
    print (u"Best Round: %d" % cv_result.shape[0])
    print (u"Detail of Best Round :" )
    print (cv_result[cv_result.shape[0] - 1:])
    return cv_result
    
def model_fit(bst, train, test, features,cv_result):
    bst.set_params(n_estimators = result.shape[0])
    
    #用训练集拟合模型
    bst.set(train[features],train[label],eval_metric = ['rmse'])
    
    #预测训练集
    train_predict= bst.predict(train[features])[:,1]
    train_rmse = metrics.rmse(train[label],train_predict)
    print ("RMSE: %f" % train_rmse)
    
    test['label1'] = bst.predict(test[features])[:,1]
    test_rmse = metrics.rmse(test[label],test['label1'])
    print ("RMSE test: %f" % test_rmse)
