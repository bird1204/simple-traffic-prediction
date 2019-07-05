# -*- coding: UTF-8 -*-
# 用淡旺季(Y)來預測流量(X)，結果很糟
# 但因為這邊只示範用 numPy, Pandas 做 linear regression 的過程，就不繼續 tune 下去了
# PS: 是說本來就沒有人會用這麼粗淺方式來預測就是了
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dataset = pd.read_csv('traffics.csv', usecols=[0, 1, 2], sep=",", skipinitialspace=True, parse_dates=['date'])
# dataset.head()
# dataset.tail()

# 檢查有沒有壞掉的資料
dataset.isna().sum()

# 清理資料：
# 1. 2016/12/31 之前把錯誤資料填上 0 ; 2017 開始，把錯誤資料刪掉
# 2. 把 Avg. page time 改成秒數
# 3. 每年的 2/15 ~ 4/30 是 peak season
# 4. Page view 都轉成 interger
# 5. Page view / avg_page_second 轉成 0 - 1

def time_convert(x):
    h,m,s = map(int,x.split(':'))
    return (h*60+m)*60+s

def page_view_convert(x):
  return int(str(x).replace(',',''))

def scale_pv(x):
  return float(x) / float(notnull_dataset['pv_int'].max())

def scale_avg(x):
  return float(x) / float(notnull_dataset['avg_page_second'].max())

older_dataset = dataset[dataset['date'] <= '2016/12/31']
older_dataset = older_dataset.fillna({'Page view': 0, 'Avg. page time': '00:00:00'})
younger_dataset = dataset[dataset['date'] >= '2017/1/1']
younger_dataset = younger_dataset.dropna()
notnull_dataset = older_dataset.append(younger_dataset)
notnull_dataset['avg_page_second'] = notnull_dataset['Avg. page time'].apply(time_convert)
notnull_dataset['is_peak_season'] = notnull_dataset['date'].dt.strftime('%m/%d').between('02/15','04/30').astype('float')
notnull_dataset['pv_int'] = notnull_dataset['Page view'].apply(page_view_convert)
notnull_dataset['pv_scale_float'] = notnull_dataset['pv_int'].apply(scale_pv)
notnull_dataset['time_scale_float'] = notnull_dataset['avg_page_second'].apply(scale_avg)

notnull_dataset.isna().sum()
notnull_dataset.describe()

# 分配資料
train_set = notnull_dataset[1800:2100]
test_set = notnull_dataset[2100:]


X = train_set[['date']].values
A = train_set[['pv_scale_float']].values
B = train_set[['is_peak_season']].values
C = train_set[['time_scale_float']].values
print('mean of pv_scale_float is',np.mean(A),'\n')
print('median of pv_scale_float is',np.median(A),'\n')

# 畫圖看一下 train_set 的長相
plt.figure()
plt.xlabel('date')
plt.subplot(4, 1, 1)
plt.subplots_adjust(hspace=1)
plt.plot(X, A, label='pv_int')
plt.plot(X, B, label='is_peak_season')
plt.plot(X, C, label = 'time_scale_float')

def reg(x,y):
    coefficients = np.polyfit(x,y,1) 
    p = np.poly1d(coefficients)
    yvals = p(x)
    return coefficients, yvals, p

# 看季節跟流量的趨勢, 並找到 y=ax+b
(arg1, arg2), yvals, formula = reg(B.flatten(), A.flatten()) 
plt.subplot(4, 1, 2)
plt.subplots_adjust(hspace=1)
plt.title("trained formula: " + str(formula), fontsize=8)
plt.plot(B.flatten(), A.flatten(), label='original')
plt.plot(B.flatten(), yvals, 'r',label='polyfit')

# 用 train data 得到的公司，測試
X = train_set[['pv_scale_float']].values
Y = train_set[['is_peak_season']].values
yvals = formula(X.flatten())
plt.subplot(4, 1, 3)
plt.subplots_adjust(hspace=1)
plt.title("test data by: " + str(formula), fontsize=8)
plt.plot(X.flatten(), Y.flatten(), label='original')
plt.plot(X.flatten(), yvals, 'r',label='polyfit')

(arg1, arg2), yvals, formula = reg(X.flatten(), Y.flatten()) 
plt.subplot(4, 1, 4)
plt.subplots_adjust(hspace=1)
plt.title("trained test formula: " + str(formula), fontsize=8)
plt.plot(X.flatten(), Y.flatten(), label='original')
plt.plot(X.flatten(), yvals, 'r',label='polyfit')

plt.legend()
plt.show()