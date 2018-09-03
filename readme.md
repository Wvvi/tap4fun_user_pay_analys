
## DataCastle 游戏玩家付费金额预测

* 此项目是DataCastle数据竞赛平台上的一个比赛

* 任务：建立模型通过用户注册账户后7天之内的游戏数据，预测用户在45天内的消费金额

* (1)训练集（带标签）：2288007个样本 带标签的训练集中共有2288007个样本。tap_fun_train.csv中存有训练集样本的全部信息，user_id为样本的id，prediction_pay_price为训练集标签，其他字段为特征。

* (2)测试集：828934个样本 tap_fun_test.csv中存有测试集的特征信息，除无prediction_pay_price字段外，格式同tap_fun_train.csv。参赛者的目标是尽可能准确地预测第45天的消费金额prediction_pay_price。

  #  前期做下探索性数据分析，了解游戏下数据、


```python
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
train_df=pd.read_csv('tap4fun_compitition_data/tap_fun_train.csv',sep=',',usecols=[105,106,107,108])
train_df.head()
```

![train_df](F:\gitrepository\tap4fun_user_pay_analys\pictures\train_df.png)






```python
#整体分析
train_df['pay_diff']=train_df['prediction_pay_price']-train_df['pay_price']
#前7天支付，7—45继续支付
index=train_df[(train_df['pay_price']>0) & (train_df['pay_diff']>0)].index
train_df.loc[index,'price_tag']='7_pay_45_pay'
#前7天支付，7-45不再支付
index=train_df[(train_df['pay_price']>0) & (train_df['pay_diff']==0)].index
train_df.loc[index,'price_tag']='7_pay_45_nopay'
#前7天不支付，7-45天支付
index=train_df[(train_df['pay_price']==0) & (train_df['pay_diff']>0)].index
train_df.loc[index,'price_tag']='7_nopay_45_pay'
#前7天不支付，7-45也不之付
index=train_df[(train_df['pay_price']==0) & (train_df['pay_diff']==0)].index
train_df.loc[index,'price_tag']='7_nopay_45_nopay'

price_tag_dummies=pd.get_dummies(train_df['price_tag'])
train_df=train_df.join(price_tag_dummies)
train_df.head()
```

![train_dummies](F:\gitrepository\tap4fun_user_pay_analys\pictures\train_dummies.png)




```python
pay_df=DataFrame()
cols=price_tag_dummies.columns
for col in cols:
    pay_df.loc[col,'people_cnt']=len(train_df[train_df[col]==1])
    pay_df.loc[col,'people_cnt_ratio']=round(len(train_df[train_df[col]==1])/len(train_df),3)
    pay_df.loc[col,'7_pay']=train_df[train_df[col]==1]['pay_price'].sum()
    pay_df.loc[col,'7_pay_ratio']=round(train_df[train_df[col]==1]['pay_price'].sum()/train_df['pay_price'].sum(),3)
    pay_df.loc[col,'7_45_pay']=train_df[train_df[col]==1]['pay_diff'].sum()
    pay_df.loc[col,'7_45_pay_ratio']=round(train_df[train_df[col]==1]['pay_diff'].sum()/train_df['pay_diff'].sum(),3)
    pay_df.loc[col,'total_pay']=train_df[train_df[col]==1]['prediction_pay_price'].sum()
    pay_df.loc[col,'total_pay_ratio']=round(train_df[train_df[col]==1]['prediction_pay_price'].sum()/train_df['prediction_pay_price'].sum(),3)
pay_df.loc['total',:]=pay_df.apply(np.sum,axis=0).values
pay_df
```

![pay_df](F:\gitrepository\tap4fun_user_pay_analys\pictures\pay_df.png)

* 通过分析训练集的付费情况可以看出,总人数224万人,付费人数45988,比例只有2%,比重非常的小,前7天付费的占总人数0.2%,后45天付费的占总人数的1.8%


```python
#只分析付费玩家
pay_new_df=pay_df.drop(['7_nopay_45_nopay','total'],axis=0)
pay_new_df['people_cnt_ratio']=pay_new_df['people_cnt']/pay_new_df['people_cnt'].sum()
pay_new_df['7_pay_ratio']=pay_new_df['7_pay']/pay_new_df['7_pay'].sum()
pay_new_df['total_pay_ratio']=pay_new_df['total_pay']/pay_new_df['total_pay'].sum()
pay_new_df.loc['total',:]=pay_new_df.apply(np.sum,axis=0).values
```


```python
pay_new_df
```

![pay_new_df](F:\gitrepository\tap4fun_user_pay_analys\pictures\pay_new_df.png)



* 在付费的人数中,前7天付费而后45天却没有付费的人数的比重最大,约66%，总额占前7天付费总额的28%,前7天付费，后45天付费的人数占付费人数的24%,不过其前7天的消费总额占前7天付费总额的70%,其后45天付费的总额占后45天付费总额的87%。不过大额的付费数据还是后45天

* 通过统计分析,我们可以大概的制定这样的策略：针对前7天付费的玩家先进行分类，得到后45天会继续付费的玩家，接着用回归预测出后45天付费的金额


##  文件说明

* dataframe_memory_optimization.py 用于减小文件所占用的内存

* common_model.py 公共函数工具

* tap4fun_analys.py 训练模型与预测

* search_best_model.py 网格法寻找最优参数

* feature_selector.py 一位外国大神写的特征选择的工具，链接：https://github.com/WillKoehrsen/feature-selector

  # 环境

* python3

* lightgbm

* matplotlib

* pandas

* scikit-learn

  ​

  ​

  ​