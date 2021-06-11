import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats


df_test = pd.read_csv("../../test.csv")
df_train = pd.read_csv("../../train.csv")

#查看目标数据详细信息
#count      1460.000000
#mean     180921.195890
#std       79442.502883
#min       34900.000000
#25%      129975.000000
#50%      163000.000000
#75%      214000.000000
#max      755000.000000
df_train['SalePrice'].describe();

#查看目标数据的
sns.distplot(df_train['SalePrice'])
#plt.show()

#查看所有数据的相关系数
corrmat = df_train.corr()
f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=1,vmin= -1,square=True)
#plt.show()

#查看正相关性最强的前10个值
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
#plt.show();

#对缺失值的处理
#查看数据内所有缺失的信息
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#对于缺失值，不同情况不同分析  高特征的低缺失值可以尝试填充估计；高缺失值的可以通过回归估计计算
#低特征的低缺失值可以不做处理；高缺失值的可直接剔除字段
#通过观察发现出现缺失值的字段的相关系数都很低，特征都不明显，因此可以删除
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

#观测离群值
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#删除离群点 提高准确度
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#方差齐次检验
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

#正偏系数大于一，高峰偏左，长尾向右延伸，处于正偏态
#进行对数变换（log transformation）让数据符合假设
df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

#也呈现正偏态，进行对数变换
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

#TotalBsmtSF 内部数据存在0值，无法进行归一化
#df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
#df_train['HasBsmt'] = 0
#df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1 #进行切片，将totalbsmtsf 大于1 的都提出 hasbsmt赋值1 其余=0 的赋值为0

#对那些做了标签的项，即TotalBsmtSF>0 的项 进行对数变换
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

df_train = pd.get_dummies(df_train)

#print(df_train)

pd.DataFrame(df_train).to_csv('../../CleantrainingDataSet_Bang.csv', index=False)


