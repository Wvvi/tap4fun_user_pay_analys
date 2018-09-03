# usr/bin/python3
# coding:utf8
import sys
sys.path.append('../feature-selector/')
from feature_selector import FeatureSelector
import pandas as pd
import numpy as np
import pickle
import gc
from pandas import DataFrame, Series
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from common_model import cross_validation, train_predict, score_models
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class handle:
    def __init__(self):
        self.train_data_classification = 'train_data_classification.csv'
        self.train_data_regression = 'train_data_regression.csv'
        self.feature_df = None
        self.Ids = None
        self.label_df = None
        self.drop_columns = None
        self.clf_reg = LinearRegression()
        self.clf_classifier = GradientBoostingClassifier()
        self.CV = KFold(n_splits=10)
        # self.CV=ShuffleSplit(n_splits=10,test_size=.2,random_state=0)

    def read_data(self, file_path):
        data = pd.read_csv(file_path, sep=',')
        print(data.head())
        self.feature_df = data.drop('pay_diff', axis=1)
        self.label_df = data['pay_diff']
        del data
        gc.collect()

    def get_models(self, reg=None):
        if reg:
            # gb_reg = GradientBoostingRegressor(
            #    n_estimators=100, max_features='sqrt', subsample=0.8, random_state=0)
            # rf_reg = RandomForestRegressor(
            #    n_estimators=100, max_features='sqrt', random_state=0)
            lr = LinearRegression()
            models = {
                # 'gb_reg': gb_reg,
                # 'rf_reg': rf_reg,
                'lr': lr}
        else:
            gb_classifier = GradientBoostingClassifier(
                subsample=0.8, n_estimators=100, random_state=10)
            # rf_classifier = RandomForestClassifier(
            #    n_estimators=10, max_features='sqrt', random_state=10)
            models = {
                'gb_classifier': gb_classifier
                # 'rf_classifier': rf_classifier
            }
        return models

    def print_model_score(self, model_dict, reg=None, scaler=None):
        # 输出模型评估报告
        X_train, y_train, X_test, y_test, cross_report = cross_validation(
            model_dict, self.feature_df, self.label_df, self.CV, reg, scaler)
        P = train_predict(model_dict, X_train, y_train, X_test, reg)
        models_df = score_models(P, y_test, reg)
        print(models_df)

    def get_best_params(self, reg=None):
        #寻找分类的最优参数
        self.read_data(self.train_data_classification)
        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        models = self.get_models()
        # X_train, y_train, X_test, y_test, cross_report = cross_validation(
        #    models, self.feature_df, self.label_df, self.CV, reg, scaler)
        # self.print_model_score(models)
        param_grid = {
            'n_estimators': range(3100, 4001, 100),
            # 'learing_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            # 'max_features': range(2, 10, 2),
            # 'max_depth': range(3, 10, 1),
            # 'min_samples_split':range(100,1001,100),
            # 'min_samples_leaf': range(50, 201, 10)
        }
        #param_grid = [{'n_estimators': range(50, 201, 10)}]
        gbm = GradientBoostingClassifier(subsample=0.8, warm_start=True, random_state=10,
                                         # n_estimators=130,
                                         learning_rate=0.001,
                                         max_features=6,
                                         max_depth=4,
                                         min_samples_split=100,
                                         min_samples_leaf=180
                                         )
        gsearch = GridSearchCV(
            gbm, param_grid, cv=5, scoring='roc_auc', n_jobs=4, iid=False)
        #gsearch.fit(X_train, y_train)
        gsearch.fit(self.feature_df, self.label_df)
        best_model = gsearch.best_estimator_
        print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
        models = {'gbm_best': best_model}
        self.print_model_score(models)

    def get_best_regression_params(self):
        #寻找回归的最优参数
        self.read_data(self.train_data_regression)
        model_lgb = lgb.LGBMRegressor(boosting_type='gbdt',
                                      num_leaves=20,
                                      max_depth=6,
                                      # learning_rate=0.01,
                                      n_estimators=4400,
                                      # subsample_for_bin=200000,
                                      # class_weight=None,
                                      # min_split_gain=0.0,
                                      min_child_weight=0.001,
                                      min_child_samples=29,
                                      subsample=0.8,
                                      metric='rmse',
                                      bagging_fraction=0.6,
                                      feature_fraction=0.8,
                                      #subsample_freq = 0,
                                      #colsample_bytree = 1.0,
                                      reg_alpha=0.5,
                                      reg_lambda=0.0,
                                      random_state=100,
                                      n_jobs=4,
                                      silent=True,
                                      importance_type='split',
                                      )
        params_cv = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'learning_rate': 0.1,
            'num_leaves': 50,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        params_grid = {
            'learn_rate': [0.1, 0.015, 0.018, 0.02]
            # 'max_depth': range(4, 9, 1),
            # 'num_leaves': range(10, 41, 1)
            # 'min_child_samples':range(15,31,1),
            # 'min_child_weight':[0.001,0.002]
            # 'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
            # 'bagging_fraction':[0.6,0.7,0.8,0.9,1.0]
            # 'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
            # 'reg_lambda':[0,0.001,0.01,0.03,0.08,0.3,0.5]

        }
        data_train = lgb.Dataset(self.feature_df, self.label_df, silent=True)
        #  cv_results = lgb.cv(params_cv, data_train, num_boost_round=1000,nfold=10,
        #  stratified=False, shuffle=True, metrics='rmse',
        #  early_stopping_rounds=50, verbose_eval=50, show_stdv=True,seed=0)
        #print('best n_estimators:', len(cv_results['rmse-mean']))
        #print('best cv score:', cv_results['rmse-mean'][-1])

        gsearch = GridSearchCV(model_lgb, params_grid,
                               scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
        gsearch.fit(self.feature_df, self.label_df)
        # model_lgb.fit(self.feature_df,self.label_df)
        best_model = gsearch.best_estimator_
        best_model = model_lgb
        #print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
        models = {'lgb_reg_best': best_model}
        self.print_model_score(models, reg=True)
        reg_model = joblib.load('lgb_reg_best.pkl')
        print(reg_model)


if __name__ == '__main__':
    H = handle()
    # H.get_best_params()
    H.get_best_regression_params()
