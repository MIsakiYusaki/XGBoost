# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:41:33 2020

@author: Yuki
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#载入文件
train = pd.read_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train.csv') 
test = pd.read_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\test.csv')
populationData = pd.read_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\datasets_507962_1091873_population_by_country_2020.csv')

pd.options.display.max_columns = 30

#处理缺失值
train['Province_State'].fillna('None',inplace=True)
train['County'].fillna("None",inplace=True)
population[population.columns].fillna("0",inplace = True)
train.drop('Id',inplace=True, axis=1)
train.head(10)

#Test Data
test['Province_State'].fillna('None',inplace=True)
test['County'].fillna("None",inplace=True)

test.drop('ForecastId',inplace=True, axis=1)
test

#分离确诊和死亡
train_Confirmed = train[train['Target']=='ConfirmedCases']
train_Fatalities = train[train['Target']=='Fatalities']
train_Confirmed.rename(columns={'TargetValue':'ConfirmedCases'},inplace = True)
train_Fatalities.rename(columns={'TargetValue':'Fatalities'},inplace = True)

#Test Data
test_Confirmed = test[test['Target']=='ConfirmedCases']
test_Fatalities = test[test['Target']=='Fatalities']
test_Confirmed.rename(columns={'TargetValue':'ConfirmedCases'},inplace = True)
test_Fatalities.rename(columns={'TargetValue':'Fatalities'},inplace = True)


#合并数据框前前预处理

train_Confirmed.drop('Target',axis=1,inplace = True)
train_Confirmed
train_Fatalities.drop('Target',axis=1,inplace = True)
train_Fatalities

#Test Data
test_Confirmed.drop('Target',axis=1,inplace = True)
test_Confirmed
test_Fatalities.drop('Target',axis=1,inplace = True)
test_Fatalities



#合并数据框
train_modified = pd.merge(train_Confirmed,train_Fatalities,on = ['County','Province_State','Country_Region','Population','Date'])
#train_modified.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train_modified.csv')

#

test_modified = pd.merge(test_Confirmed,test_Fatalities,on = ['County','Province_State','Country_Region','Population','Date'])
#不要以下四个地方的数据  


dropvalues = ('Diamond Princess', 'Kosovo', 'MS Zaandam', 'West Bank and Gaza','Andorra')
#b1 = train_modified['Country_Region'] == dropvalues ##产生布尔值

for v in dropvalues:
    train_modified = train_modified.drop(index = (train_modified.loc[(train_modified['Country_Region']==v)].index))

#Test Data
for v in dropvalues:
    test_modified = test_modified.drop(index = (test_modified.loc[(test_modified['Country_Region']==v)].index))



#验证某日全球新增确诊人数
df1 = train_Confirmed[train['Date']=='2020-01-24']
df1['ConfirmedCases'].sum()

#date = train_Confirmed['Date']

##用累加法得出累计人数
confirmed_total_date = np.cumsum(train_modified.groupby(['Date']).agg({'ConfirmedCases':['sum']}))
fatalities_total_date = np.cumsum(train_modified.groupby(['Date']).agg({'Fatalities':['sum']}))
total_date = confirmed_total_date.join(fatalities_total_date)
total_date
#绘图得出人数
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date.plot(ax=ax1)
ax1.set_title("Global confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date.plot(ax=ax2, color='orange')
ax2.set_title("Global deceased cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)

#重建Id列
train_modified['Id'] = np.arange(len(train_modified['Date']))
train_modified
#Test Data

test_modified['ForecastId'] = np.arange(len(test_modified['Date']))
test_modified
#train_Confirmed.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\Confirmed_train.csv')
#train_Fatalities.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\Fatalities_train.csv')

#验证是否存在没有详细数据的国家

#set(train_modified['Country_Region'].unique()).difference(set(population['Country (or dependency)'].unique()))

#全球趋势（除中国）

confirmed_total_date_noChina = np.cumsum(train_modified[train_modified['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']}))
fatalities_total_date_noChina = np.cumsum(train_modified[train_modified['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']}))
total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date_noChina.plot(ax=ax1)
ax1.set_title("Global confirmed cases excluding China", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date_noChina.plot(ax=ax2, color='orange')
ax2.set_title("Global deceased cases excluding China", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)

#中国趋势

confirmed_total_date_China = np.cumsum(train_modified[train_modified['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']}))
fatalities_total_date_China = np.cumsum(train_modified[train_modified['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']}))
total_date_China = confirmed_total_date_China.join(fatalities_total_date_China)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date_China.plot(ax=ax1)
ax1.set_title("China confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date_China.plot(ax=ax2, color='orange')
ax2.set_title("China deceased cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)

##四国对比

confirmed_total_date_Italy = train_modified[(train_modified['Country_Region']=='Italy') & train_modified['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Italy = train_modified[(train_modified['Country_Region']=='Italy') & train_modified['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)

confirmed_total_date_Spain = train_modified[(train_modified['Country_Region']=='Spain') & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Spain = train_modified[(train_modified['Country_Region']=='Spain') & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)

confirmed_total_date_UK = train_modified[(train_modified['Country_Region']=='United Kingdom') & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_UK = train_modified[(train_modified['Country_Region']=='United Kingdom') & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_UK = confirmed_total_date_UK.join(fatalities_total_date_UK)

confirmed_total_date_Singapore = train_modified[(train_modified['Country_Region']=='Singapore') & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_Singapore = train_modified[(train_modified['Country_Region']=='Singapore') & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_Singapore = confirmed_total_date_Singapore.join(fatalities_total_date_Singapore)

italy = [i for i in np.cumsum(total_date_Italy.ConfirmedCases['sum']).values]

italy_70 = italy[0:70] 
spain = [i for i in np.cumsum(total_date_Spain.ConfirmedCases['sum']).values]
spain_70 = spain[0:70] 
UK = [i for i in np.cumsum(total_date_UK.ConfirmedCases['sum']).values]
UK_70 = UK[0:70] 


singapore = [i for i in np.cumsum( total_date_Singapore.ConfirmedCases['sum']).values]
singapore_70 = singapore[0:70] 

plt.figure(figsize=(12,6))
plt.plot(italy_70)
plt.plot(spain_70)
plt.plot(UK_70)
plt.plot(singapore_70)
plt.legend(["Italy", "Spain", "UK", "Singapore"], loc='upper left')
plt.title("COVID-19 infections from the first confirmed case", size=15)
plt.xlabel("Days", size=13)
plt.ylabel("Infected cases", size=13)
plt.ylim()
plt.show()

#体外实验
list1 = ["v","one","tyt"]
Country = pd.DataFrame("None",columns = list1,index = list(np.arange(70)))
Country
arr1 = np.array([i for i in np.cumsum(total_date_UK.ConfirmedCases['sum']).values])[0:70]
arr1.size
arr1.shape
Country['v'] = arr1
Country
np.cumsum(total_date_UK.ConfirmedCases['sum'].values)[0:70]


# def CountryContrast(v):
    
#     Country = pd.DataFrame("None",columns = v,index = list(np.arange(70)))
#     for c in v :
#         confirmed_total_date = train_modified[(train_modified['Country_Region']== c) & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})
#         fatalities_total_date = train_modified[(train_modified['Country_Region']==c) & (train_modified['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})
#         total_date = confirmed_total_date.join(fatalities_total_date)
#         arr1 = -np.cumsum(total_date.ConfirmedCases['sum'].values)[0:70]
#         #arr2 = arr1[:]
#         #arr2 = np.random.rand(70)
#     #return arr1,arr2,len(arr2)
#         Country[c]= arr1
        

#     return Country
    
#     plt.figure(figsize=(12,6))
#     for d in v:
#     plt.plot(Country[d])
#     plt.legend(v, loc = 'upper left')
#     plt.title("COVID-19 infections from the first confirmed case", size=15)
#     plt.xlabel("Days", size=13)
#     plt.ylabel("Infected cases", size=13)
#     plt.ylim()
#     plt.show()
    

# Countries = ["Italy", "Spain", "UK", "Singapore"]

# CountryContrast(Countries)    
    
    
    
    
#SIR Model

# Susceptible equation
def fa(N, a, b, beta):
    fa = -beta*a*b
    return fa

# Infected equation
def fb(N, a, b, beta, gamma):
    fb = beta*a*b - gamma*b
    return fb

# Recovered/deceased equation
def fc(N, b, gamma):
    fc = gamma*b
    return fc
#数值分析方法
# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)
def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):
    a1 = fa(N, a, b, beta)*hs
    b1 = fb(N, a, b, beta, gamma)*hs
    c1 = fc(N, b, gamma)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(N, ak, bk, beta)*hs
    b2 = fb(N, ak, bk, beta, gamma)*hs
    c2 = fc(N, bk, gamma)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(N, ak, bk, beta)*hs
    b3 = fb(N, ak, bk, beta, gamma)*hs
    c3 = fc(N, bk, gamma)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(N, ak, bk, beta)*hs
    b4 = fb(N, ak, bk, beta, gamma)*hs
    c4 = fc(N, bk, gamma)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c


def SIR(N, b0, beta, gamma, hs):
    
    """
    N = total number of population
    beta = transition rate S->I
    gamma = transition rate I->R
    k =  denotes the constant degree distribution of the network (average value for networks in which 
    the probability of finding a node with a different connectivity decays exponentially fast
    hs = jump step of the numerical integration
    """
    
    # Initial condition
    a = float(N-1)/N -b0
    b = float(1)/N +b0
    c = 0.

    sus, inf, rec= [],[],[]
    for i in range(10000): # Run for a certain number of time-steps
        sus.append(a)
        inf.append(b)
        rec.append(c)
        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)

    return sus, inf, rec

#模型的参数
N = 7800*(10**6)
b0 = 0
beta = 0.7
gamma = 0.2
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title("SIR model")
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR_example.png')
plt.show()

#用真实数据拟合模型

population = float(46750238) #Spain Population
country_df = pd.DataFrame()
country_df['ConfirmedCases'] = np.abs(train_modified.loc[train_modified['Country_Region']=='Spain'].ConfirmedCases.fillna(0))
country_df = country_df[10:]
country_df['day_count'] = list(range(1,len(country_df)+1))

ydata = [i for i in country_df.ConfirmedCases]
ydata
xdata = country_df.day_count
ydata = np.array(ydata, dtype=float)
xdata = np.array(xdata, dtype=float)

N = population
inf0 = ydata[0]
sus0 = N - inf0
rec0 = 0.0

def sir_model(y, x, beta, gamma):
    sus = -beta * y[0] * y[1] / N
    rec = gamma * y[1]
    inf = -(sus + rec)
    return sus, inf, rec

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata, *popt)

plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted)
plt.title("Fit of SIR model for Spain infected cases")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()
print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])






train_modified
population



#丰富数据

train_enriched = pd.merge(train_modified,populationData,left_on=['Country_Region'],right_on=['Country (or dependency)'],how = 'left')
test_enriched = pd.merge(test_modified,populationData,left_on=['Country_Region'],right_on=['Country (or dependency)'],how = 'left')
#train_enriched[351664:351670]
train_enriched.drop(['Country (or dependency)','Population'],inplace = True,axis = 1)
test_enriched.drop(['Country (or dependency)','Population'],inplace = True,axis = 1)
train_enriched.columns
train_enriched
test_enriched

#初始化人数
test_enriched['ConfirmedCases'] = 0
test_enriched['Fatalities'] = 0
#重排
order = ['Id','Date','Country_Region','Province_State','County','Population (2020)','Yearly Change','Net Change',
         'Density (P/Km²)','Land Area (Km²)','Migrants (net)','Fert. Rate','Med. Age','Urban Pop %','World Share','ConfirmedCases','Fatalities']
order1 = ['ForecastId','Date','Country_Region','Province_State','County','Population (2020)','Yearly Change','Net Change',
         'Density (P/Km²)','Land Area (Km²)','Migrants (net)','Fert. Rate','Med. Age','Urban Pop %','World Share','ConfirmedCases','Fatalities']
train_enriched = train_enriched[order]
test_enriched = test_enriched[order1]
#输出丰富后的数据

train_enriched
test_enriched
##规范化数据
Convert = ['Yearly Change', 'Fert. Rate', 'Med. Age', 'Urban Pop %','World Share']

for v in Convert:
    train_enriched[v] = train_enriched[v].str.strip('%')
    test_enriched[v] = test_enriched[v].str.strip('%')

##去除N.A.值，此类国家没有详细国家情况，不值得参考
train_enriched = train_enriched.drop(index = (train_enriched.loc[(train_enriched['Fert. Rate']=='N.A.')].index))
train_enriched = train_enriched.drop(index = (train_enriched.loc[(train_enriched['Urban Pop %']=='N.A.')].index))

test_enriched = test_enriched.drop(index = (test_enriched.loc[(test_enriched['Fert. Rate']=='N.A.')].index))
test_enriched = test_enriched.drop(index = (test_enriched.loc[(test_enriched['Urban Pop %']=='N.A.')].index))
test_enriched

#合并训练集和测试集

all_data = pd.concat([train_enriched,test_enriched], axis = 0, sort = False)
all_data
all_data.loc[all_data['Date'] >= '2020-04-01', 'ConfirmedCases'] = 0
all_data.loc[all_data['Date'] >= '2020-04-01', 'Fatalities'] = 0
all_data['Date'] = pd.to_datetime(all_data['Date'])

# Create date columns
le = preprocessing.LabelEncoder()
all_data['Day_num'] = le.fit_transform(all_data.Date)
all_data['Day'] = all_data['Date'].dt.day
all_data['Month'] = all_data['Date'].dt.month
all_data['Year'] = all_data['Date'].dt.year

# Fill null values given that we merged train-test datasets
all_data['Province_State'].fillna("None", inplace=True)
all_data['ConfirmedCases'].fillna(0, inplace=True)
all_data['Fatalities'].fillna(0, inplace=True)
all_data['Id'].fillna(-1, inplace=True)
all_data['ForecastId'].fillna(-1, inplace=True)

display(all_data)
display(all_data.loc[all_data['Date'] == '2020-05-01'])
#统计缺失值
missings_count = {col:all_data[col].isnull().sum() for col in all_data.columns}
missings = pd.DataFrame.from_dict(missings_count, orient='index')
print(missings.nlargest(30, 0))



##计算延迟

def calculate_lag(df, lag_list, column):
    for lag in lag_list:
        column_lag = column + "_" + str(lag)
        df[column_lag] = df.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0)
    return df

def calculate_trend(df, lag_list, column):
    for lag in lag_list:
        trend_column_lag = "Trend_" + column + "_" + str(lag)
        df[trend_column_lag] = (df.groupby(['Country_Region', 'Province_State'])[column].shift(0, fill_value=0) - 
                                df.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0))/df.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0.001)
    return df

ts = time.time()
all_data = calculate_lag(all_data.reset_index(), range(1,7), 'ConfirmedCases')
all_data = calculate_lag(all_data, range(1,7), 'Fatalities')
all_data = calculate_trend(all_data, range(1,7), 'ConfirmedCases')
all_data = calculate_trend(all_data, range(1,7), 'Fatalities')
all_data.replace([np.inf, -np.inf], 0, inplace=True)
all_data.fillna(0, inplace=True)
print("Time spent: ", time.time()-ts)

all_data[all_data['Country_Region']=='Spain'].iloc[40:50][['Id', 'Province_State', 'Country_Region', 'Date',
       'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num', 'ConfirmedCases_1',
       'ConfirmedCases_2', 'ConfirmedCases_3', 'Fatalities_1', 'Fatalities_2',
       'Fatalities_3']]


all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])
number_c = all_data['Country_Region']
countries = le.inverse_transform(all_data['Country_Region'])
country_dict = dict(zip(countries, number_c)) 
all_data['Province_State'] = le.fit_transform(all_data['Province_State'])
number_p = all_data['Province_State']
province = le.inverse_transform(all_data['Province_State'])
province_dict = dict(zip(province, number_p))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

# 
y1 = np.cumsum(all_data[(all_data['Country_Region']==country_dict['Spain']) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)][['ConfirmedCases']])
x1 = range(0, len(y1))

ax1.plot(x1, y1)
ax1.set_title("Spain ConfirmedCases between days 69 and 79")
ax1.set_xlabel("Days")
ax1.set_ylabel("ConfirmedCases")

y2 = np.cumsum(all_data[(all_data['Country_Region']==country_dict['Spain']) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)][['ConfirmedCases']].apply(lambda x: np.log(x)))
x2 = range(0, len(y2))
ax2.plot(x2, y2, 'bo--')
ax2.set_title("Spain Log ConfirmedCases between days 69 and 79")
ax2.set_xlabel("Days")
ax2.set_ylabel("Log ConfirmedCases")

##
data = all_data.copy()
features = ['Id', 'ForecastId', 'Country_Region', 'Province_State', 'ConfirmedCases', 'Fatalities', 
       'Day_num']
data = data[features]
data
# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends
data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')
data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log1p(x))

# Replace infinites
data.replace([np.inf, -np.inf], 0, inplace=True)


# Split data into train/test
def split_data(df, train_lim, test_lim):
    
    df.loc[df['Day_num']<=train_lim , 'ForecastId'] = -1
    df = df[df['Day_num']<=test_lim]
    
    # Train set
    x_train = df[df.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)
    y_train_1 = df[df.ForecastId == -1]['ConfirmedCases']
    y_train_2 = df[df.ForecastId == -1]['Fatalities']

    # Test set
    x_test = df[df.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    # Clean Id columns and keep ForecastId as index
    x_train.drop('Id', inplace=True, errors='ignore', axis=1)
    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    x_test.drop('Id', inplace=True, errors='ignore', axis=1)
    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)
    
    return x_train, y_train_1, y_train_2, x_test


def lin_reg(X_train, Y_train, X_test):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)
    
    return regr, y_pred


# Submission function
def get_submission(df, target1, target2):
    
    prediction_1 = df[target1]
    prediction_2 = df[target2]

    # Submit predictions
    prediction_1 = [int(item) for item in list(map(round, prediction_1))]
    prediction_2 = [int(item) for item in list(map(round, prediction_2))]
    
    submission = pd.DataFrame({
        "ForecastId": df['ForecastId'].astype('int32'), 
        "ConfirmedCases": prediction_1, 
        "Fatalities": prediction_2
    })
    submission.to_csv('submission.csv', index=False)
    
dates_list = ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', 
                 '2020-03-10', '2020-03-11','2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',
                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', 
                 '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05', 
                 '2020-04-06', '2020-04-07', '2020-04-08', '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13', '2020-04-14']    

all_data.loc[all_data['Country_Region']==country_dict['Spain']][50:70]
all_data
def plot_linreg_basic_country(data, country_name, dates_list, day_start, shift, train_lim, test_lim):
    
    data_country = data[data['Country_Region']==country_dict[country_name]]
    data_country = data_country.loc[data_country['Day_num']>=day_start]
    X_train, Y_train_1, Y_train_2, X_test = split_data(data_country, train_lim, test_lim)
    model, pred = lin_reg(X_train, Y_train_1, X_test)

    # Create a df with both real cases and predictions (predictions starting on March 12th)
    X_train_check = X_train.copy()
    X_train_check['Target'] = Y_train_1

    X_test_check = X_test.copy()
    X_test_check['Target'] = pred

    X_final_check = pd.concat([X_train_check, X_test_check])

    # Select predictions from March 1st to March 25th
    predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target
    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']
    dates_list_num = list(range(0,len(dates_list)))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

    ax1.plot(dates_list_num, np.expm1(predicted_data))
    ax1.plot(dates_list_num, real_data)
    ax1.axvline(30-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)
    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    ax1.set_xlabel("Day count (from March " + str(1+shift) + " to March 25th)")
    ax1.set_ylabel("Confirmed Cases")

    ax2.plot(dates_list_num, predicted_data)
    ax2.plot(dates_list_num, np.log1p(real_data))
    ax2.axvline(30-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)
    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')
    ax2.set_xlabel("Day count (from March " + str(1+shift) + " to March 30th)")
    ax2.set_ylabel("Log Confirmed Cases")

    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
    
    
data    
# Filter Spain, run the Linear Regression workflow
country_name = "Spain"
march_day = 10
day_start = 39+march_day
dates_list2 = dates_list[march_day:]
train_lim, test_lim = 69, 112
plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day, train_lim, test_lim)




#将字符串类型的列转换为浮点数
for v in Convert:
    train_enriched[v] = train_enriched[v].astype(float)

#将日期转化为从2020-01-23开始后的天数

train_enriched['Date'] = train_enriched['Date'].astype('category').values.codes
train_enriched['Date'] = train_enriched['Date'] + 1
train_enriched.rename(columns={'Date':'Days'},inplace = True)
train_enriched['Days'] = train_enriched['Days'].astype(int)
train_enriched['Days']
train_enriched
#train_enriched.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train_enriched.csv')

#将分类型变量编码
train_enriched_Code = train_enriched.copy()

train_enriched_Code

train_enriched_Code['Country_Region'] = train_enriched['Country_Region'].astype('category').values.codes
train_enriched_Code['Province_State'] = train_enriched['Province_State'].astype('category').values.codes
train_enriched_Code['County'] = train_enriched['County'].astype('category').values.codes
train_enriched_Code
#Index = ['Date', 'Country_Region', 'Province_State', 'County',
   #    'Population (2020)', 'Yearly Change', 'Net Change', 'Density (P/Km²)',
   #    'Land Area (Km²)', 'Migrants (net)', 'Fert. Rate', 'Med. Age',
   #    'Urban Pop %', 'World Share', 'ConfirmedCases', 'Fatalities']

#columns = train_enriched.columns
#columns
#for v in columns:
    #train_enriched[v] = train_enriched[v].astype(float)

#train_enriched_Code.to_csv('C:\\Users\Yuki\\Desktop\\COVID-19\\covid19-global-forecasting-week-5\\train_enriched_Code.csv')
##数据预处理完成

train_enriched['Id'] = np.arange(len(train_enriched['County']))
train_enriched














#尝试使用XGBOOST进行回归
import xgboost as xgb
from sklearn.model_selection import train_test_split

variables = ['Days','Country_Region','Province_State','County','Population (2020)','Yearly Change','Net Change',
         'Density (P/Km²)','Land Area (Km²)','Migrants (net)','Fert. Rate','Med. Age','Urban Pop %','World Share']

#Target1 = ['ConfirmedCases']
#Target2 = ['Fatalities']

#x = train_enriched[variables]
#y1 = train_enriched[Target1]
#y2 = train_enriched[Target2]


#x
#y1
#y2

#x_train,x_test,y1_train,y1_test = train_test_split(x,y1,test_size = 1/5.,random_state = 8)
#x_train
#x_test
#y1_train
#y1_test

#将数据归一化便于后边进行logistics回归
#from sklearn import preprocessing
#from sklearn.metrics import roc_auc_score
#minmax = preprocessing.MinMaxScaler()

#train_enriched['ConfirmedCaseMaxMin']= minmax.fit_transform(np.array(train_enriched['ConfirmedCases']).reshape(-1,1))
#train_enriched['ConfirmedCaseMaxMin']

#train_enriched

mask = np.random.rand(len(train_enriched_Code)) < 0.8
train = train_enriched_Code[mask]
test = train_enriched_Code[~mask]


#['Date','Country_Region','Province_State','County','Population (2020)','Yearly Change','Net Change',
         #'Density (P/Km²)','Land Area (Km²)','Migrants (net)','Fert. Rate','Med. Age','Urban Pop %','World Share']

train[variables]
train['ConfirmedCases']

xgb_train = xgb.DMatrix(train[variables],label = train['ConfirmedCases'])
xgb_test = xgb.DMatrix(test[variables],label = test['ConfirmedCases'])





#xgb_train = xgb.DMatrix(x_train,label = y1_train)
#xgb_test = xgb.DMatrix(x_test,label = y1_test)
#xgb_train
#xgb_test
params = {
    "objective":"reg:linear",
    "booster":"gbtree",
    "eta":0.1,
    "min_child_weight":1,
    "max_depth":6,
    "eval_metric":"mae"
        }


num_round = 500

watchlist = [(xgb_train,'train'),(xgb_test,'test')]

model = xgb.train(params, xgb_train,num_round, watchlist,early_stopping_rounds=20)
res = xgb.cv(params,xgb_train,num_round,nfold=5,metrics={'rmse'},seed = 0,callbacks = [xgb.callback.print_evaluation(show_stdv=False),xgb.callback.early_stop(10)])

model1 = xgb.train()

#xgboost 超参数处理

from xgboost.sklearn import XGBRegressor
import sklearn.model_selection
from sklearn.model_selection import GridSearchCV

from matplotlib.pylab import rcParams


#创建model_cv函数建立模型和交叉验证

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


features = variables
modelparam = XGBRegressor(
    "objective":"reg:linear",
    "booster":"gbtree",
    "eta":0.1,
    "min_child_weight":1,
    "max_depth":6,
    "eval_metric":"rmse")    
    