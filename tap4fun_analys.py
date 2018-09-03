# usr/bin/python3
# coding:utf8
import sys
sys.path.append('../feature-selector/')
# sys.setrecursionlimit(100000)
from feature_selector import FeatureSelector
import pandas as pd
import numpy as np
import pickle
import gc
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn import metrics
from common_analys import load_file
from common_model import cross_validation, train_predict, score_models, fit_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from dataframe_memory_optimization import DtypesConvert


class handle:
    def __init__(self):
        self.read_path_train = 'tap4fun_compitition_data/tap_fun_train.csv'
        self.read_path_test = 'tap4fun_compitition_data/tap_fun_test.csv'
        self.save_path = 'tap4fun_submit.csv'
        self.data = None
        self.feature_df = None
        self.Ids = None
        self.label_df = None
        self.submit_df = DataFrame()
        self.drop_columns = None
        self.clf_reg = LinearRegression()
        self.clf_classifier = GradientBoostingClassifier()
        self.CV = KFold(n_splits=10)
        # self.CV=ShuffleSplit(n_splits=10,test_size=.2,random_state=0)

    def read_data(self, file_path, chunksize=None, usecols=None):
        if chunksize:
            reader = pd.read_csv(file_path, sep=',', chunksize=chunksize)
            data = reader.get_chunk()
        else:
            if usecols:
                data = pd.read_csv(file_path, sep=',', usecols=usecols)
            else:
                data = pd.read_csv(file_path, sep=',')
        dc = DtypesConvert(data)
        optimized_df = dc.dtype_memory()
        if usecols:
            optimized_data = optimized_df
        else:
            optimized_data = dc.dataframe_dtype_converted(
                optimized_df, file_path)
        try:
            optimized_data['pay_diff'] = optimized_data['prediction_pay_price'] - \
                optimized_data['pay_price']
        except:
            pass
        self.Ids = optimized_data['user_id']
        self.data = optimized_data.drop(['user_id', 'register_time'], axis=1)
        del data, optimized_data, optimized_df
        gc.collect()

    def get_models(self, reg=None):
        if reg:
            lr = Pipeline([('poly', PolynomialFeatures(degree=10)),
                           ('linear', LinearRegression(fit_intercept=False))])
            ridge = Pipeline([('poly', PolynomialFeatures(degree=5)),
                              ('ridge', Ridge())])
            lasso = Pipeline([('poly', PolynomialFeatures(degree=10)),
                              ('lasso', Lasso())])
            lgb_reg = joblib.load('lgb_reg_best.pkl')
            models = {
                # 'lr': lr,
                # 'ridge': ridge,
                # 'lasso': lasso
                'lgb_reg': lgb_reg
            }
        else:
            gb_classifier = GradientBoostingClassifier(subsample=0.8, warm_start=True, random_state=10,
                                                       n_estimators=5000,
                                                       learning_rate=0.001,
                                                       max_features=6,
                                                       max_depth=4,
                                                       min_samples_split=100,
                                                       min_samples_leaf=180)
            models = {'gbm_classifier': gb_classifier}

        return models

    def print_model_score(self, reg=None, scaler=None):
        # 输出模型评估报告
        models = self.get_models(reg)
        X_train, y_train, X_test, y_test, cross_report = cross_validation(
            models, self.feature_df, self.label_df, self.CV, reg, scaler)
        P = train_predict(models, X_train, y_train, X_test, reg)
        models_df = score_models(P, y_test, reg)
        print(models_df)
        print('Done.\n')

    def feature_engineering(self, x_data, y_data, train=None):
        #特征选择
        cols = x_data.columns
        # 消耗
        consume_col = cols[0:10]
        # 招募
        recruit_col = cols[10:22]
        # 加速
        acceleration_col = cols[22:32]
        # 建筑
        build_col = cols[32:48]
        # 科技
        science_col = cols[48:97]
        # pvp
        pvp_col = cols[97:103]
        # 付费
        pay_col = cols[103:106]
        # label
        # label_col = cols[108]
        if train:
            fs = FeatureSelector(data=x_data, labels=DataFrame(y_data))
            fs.identify_all(selection_params={'missing_threshold': 0.6,
                                              'correlation_threshold': 0.98,
                                              'task': 'classification',
                                              'eval_metric': 'auc',
                                              'cumulative_importance': 0.99
                                              })
            self.drop_columns = fs.ops
            with open('drop_columns.pkl', 'wb') as file:
                pickle.dump(self.drop_columns, file)
            self.feature_df = fs.remove(methods='all', keep_one_hot=False)
        else:
            drop_list = []
            for key in self.drop_columns.keys():
                for value in self.drop_columns[key]:
                    drop_list.append(value)
            self.feature_df.drop(drop_list, axis=1, inplace=True)
        print(self.drop_columns)

    def feature_train(self, data, reg=None):
        if not reg:
            # 只取前7天支付的数据,7-45天继续支付的数据作为分类的数据集
            data = data[data['pay_price'] > 0]
            self.label_df = data['pay_diff'].apply(lambda x: 1 if x > 0 else 0)
            self.feature_df = data.drop(
                ['pay_diff', 'prediction_pay_price'], axis=1)
            df = pd.concat([self.label_df, self.feature_df], axis=1)
            df.to_csv('train_data_classification.csv', sep=',')
        else:
            # 将前7天支付且7-45天支付的数据作为回归的数据集
            data = data[(data['pay_price'] > 0) & (data['pay_diff'] > 0)]
            self.label_df = data['pay_diff']
            self.feature_df = data.drop(
                ['pay_diff', 'prediction_pay_price'], axis=1)
            df = pd.concat([self.label_df, self.feature_df], axis=1)
            df.to_csv('train_data_regression.csv', sep=',')

    def feature_test(self, data, reg=None):
        if not reg:
            self.submit_df['user_id'] = self.Ids
            self.submit_df['pay_price'] = data['pay_price']
            self.feature_df = data[data['pay_price'] > 0]
        else:
            pass

    def pipline_engineering(self):
        # 分类
        self.read_data(self.read_path_train, 70000)
        self.feature_train(self.data)
        self.feature_engineering(self.feature_df, self.label_df, True)
        scaler = MinMaxScaler()
        scaler = StandardScaler()
        self.print_model_score(False, scaler)
        # 回归
        self.feature_train(self.data, True)
        self.feature_engineering(self.feature_df, self.label_df)
        self.print_model_score(True, False)
        del self.data
        gc.collect()

    def test_prediction(self):
        # 先分类
        self.read_data(self.read_path_test, 70000)
        self.feature_test(self.data)
        self.feature_engineering(self.feature_df, self.label_df)
        # 先保存好要分类数据集的index
        index = self.feature_df.index
        clf_classifier = joblib.load('gbm_classifier.pkl')
        class_label = pd.Series(clf_classifier.predict(self.feature_df))
        self.submit_df.loc[index, 'prediction_pay_price'] = class_label
        print(class_label.value_counts())
        # 保存好要用于回归数据集的index
        index = self.submit_df[self.submit_df['prediction_pay_price'] == 1].index
        # self.feature_df = self.feature_df[[
        #    'pay_price', 'pay_count', 'avg_online_minutes']].loc[index]
        self.feature_df = self.feature_df.loc[index]
        clf = joblib.load('lgb_reg.pkl')
        predictions = clf.predict(self.feature_df)
        self.submit_df.loc[index, 'prediction_pay_price'] = predictions
        self.submit_df['prediction_pay_price'] = self.submit_df['prediction_pay_price'] + \
            self.submit_df['pay_price']
        self.submit_df.drop('pay_price', axis=1, inplace=True)
        self.submit_df.fillna(0, inplace=True)
        self.submit_df.to_csv(self.save_path, sep=',',
                              index=False, encoding='UTF-8')
        # print(self.submit_df['prediction_pay_price'].value_counts()[:10])


if __name__ == '__main__':
    H = handle()
    H.pipline_engineering()
    H.test_prediction()
