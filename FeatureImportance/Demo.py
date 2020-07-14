from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from scipy import stats

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor


## --------------------------------加载iris数据------------------------------------------------##
x, y = load_iris().data, load_iris().target
feature_names, target_names = load_iris().feature_names, load_iris().target_names

Data = pd.DataFrame(x, columns=feature_names)
Data['target'] = y


## --------------------------------------K-S检验------------------------------------------------##
## K-S检验 : 判断是否满足正太分布。若pvalue > 0.05，则满足正太分布 
U = []
STD = []
ks = []
for i in Data.columns:
    u = Data[i].mean()
    std = Data[i].std()
    U.append(u)
    STD.append(std)
    ## norm：正太分布，也可以替换为其他分布
    ks.append((i,stats.kstest(Data[i], 'norm', (u, std)))) 


## --------------------------------------卡方检验------------------------------------------------##
## 卡方检验  k为重要特征数目
KBest = SelectKBest(chi2, k=2).fit_transform(x, y)
# print(X_new)
for i in range(2):
    d = Data[:1].T
    print(str(i)+': ', d[d[0]==KBest[0,i]].index)


## --------------------------------------梯度特征消除法------------------------------------------------##
## 梯度特征消除法
#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
rfe_lr = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(x, y)
# print(rfe_lr)
for i in range(2):
    d = Data[:1].T
    print(str(i)+': ', d[d[0]==rfe_lr[0,i]].index)


## --------------------------------------基模型的特征选择------------------------------------------------##
#带L1惩罚项的逻辑回归作为基模型的特征选择   
Lr1 = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(x, y)
for i in range(7):
    d = Data[:1].T
    print(str(i)+': ', d[d[0]==Lr1[0,i]].index)

#带L2惩罚项的逻辑回归作为基模型的特征选择   
Lr2 = SelectFromModel(LogisticRegression(penalty="l2", C=0.1)).fit_transform(x, y)
for i in range(7):
    d = Data[:1].T
    print(str(i)+': ', d[d[0]==Lr2[0,i]].index)


## --------------------------------------随机森林------------------------------------------------##
## 随机森林
from sklearn.ensemble import RandomForestRegressor
names = feature_names
rf = RandomForestRegressor(n_estimators=20, max_depth=2)
scores = []
for i in range(x.shape[1]-2):
    score = cross_val_score(rf, Data.iloc[:, i+1:i+2], y, scoring='r2', cv=ShuffleSplit(len(x), 3, 0.3))
    scores.append((round(np.mean(score), 3), names[i]))


## --------------------------------------信息熵------------------------------------------------##
#定义计算信息熵的函数：计算Infor(D)
def infor(data):
    a = pd.value_counts(data) / len(data)
    return sum(np.log2(a) * a * (-1))

#定义计算信息增益的函数：计算g(D|A)
def g(data,str1,str2):
    e1 = data.groupby(str1).apply(lambda x:infor(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    #计算Infor(D|A)
    e2 = sum(e1 * p1)
    return infor(data[str2]) - e2

#定义计算信息增益率的函数：计算gr(D,A)
def gr(data,str1,str2):
    return g(data,str1,str2)/infor(data[str1])

gr_scores = []
for i in feature_names:
    try:
        s = round(gr(Data, i, 'target'), 5)
        gr_scores.append((i,s))
    except:
        pass
print(sorted(gr_scores))

