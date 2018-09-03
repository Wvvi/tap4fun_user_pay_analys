# usr/bin/python3
# coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from pandas import DataFrame as DF
from pandas import Series
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.externals import joblib


def cross_validation(model_list, feature_df, label_series, CV,
                     reg=None, scaler=None):
    if scaler:
        x_scaler = scaler
        feature_df = DF(x_scaler.fit_transform(feature_df))
    kf = CV
    split_df = kf.split(feature_df, label_series)
    cv_score_df = DF()
    print('cross validation....')
    #  for name, clf in model_list.items():
        #  if reg:
            #  cv_score = -(cross_val_score(clf, feature_df, label_series,
                                         #  cv=kf, scoring='neg_mean_squared_error'))
        #  else:
            #  cv_score = cross_val_score(
                #  clf, feature_df, label_series, cv=10, scoring='roc_auc')
        #  cv_score_df.loc[name, 'mean_cv_score'] = round(np.mean(cv_score), 3)
        #  cv_score_df.loc[name, 'std_cv_score'] = round(np.std(cv_score), 3)
        #  cv_score_df.loc[name, 'min_cv_score'] = round(np.min(cv_score), 3)
        #  cv_score_df.loc[name, 'max_cv_score'] = round(np.max(cv_score), 3)
    #  print(cv_score_df)
    report_df = DF()
    for train_index, test_index in split_df:
        X_train, X_test = feature_df.iloc[train_index], feature_df.iloc[test_index]
        y_train, y_test = label_series.iloc[train_index], label_series.iloc[test_index]
        df = DF(y_train.value_counts()).iloc[:3]
        df['percent(%)'] = (df/df.sum()).apply(lambda x: round(x*100, 3))
        report_df = pd.concat([report_df, df], axis=1)
    del feature_df, label_series
    gc.collect()
    print(report_df.head())
    return X_train, y_train, X_test, y_test, report_df


def fit_model(model_list, x_train, y_train):
    for name, clf in model_list.items():
        clf.fit(x_train, y_train)
        joblib.dump(clf, name+'.pkl')


def classification_model_report(clf, x_test, y_test, cv_folds, show_feature_importance=True):
    # 预测
    preditions = clf.predict(x_test)
    # 针对分类问题
    train_predprod = clf.predict_proba(x_test)[:, 1]
    print('\nModel Report')
    print('Accurary:%.4' % metrics.accuracy_score(y_test, predictions))
    print('AUC Score %f' % metrics.roc_auc_score(y_test, train_predprob))

    if show_feature_importance:
        feature_list = x_train.columns
        feature_importance = clf.feature_importances_
        feature_importance = (feature_importance /
                              feature_importance.max())*100.0
        import_index = np.where(feature_importance > 0.7)[0]
        feat_imp = Series(feature_importance[import_index], feature_list[import_index]).sort_values(
            ascending=False)[:10]
        feat_imp.plot(kind='bar', title='Feature importances')
        plt.ylabel('Feature importance Score')
        plt.show()



def train_predict(model_dict, x_train, y_train, x_test, reg=None):
    # 用训练集拟合模型，再对测试集进行预测，获得测试集的目标数据
    P = np.zeros((x_test.shape[0], len(list(model_dict))))
    P = DF(P)
    cols = []
    for i, (name, clf) in enumerate(model_dict.items()):
        print('%s..' % name, end='', flush=False)
        clf.fit(x_train, y_train)
        joblib.dump(clf, name+'.pkl')
        if reg:
            P.iloc[:, i] = clf.predict(x_test)
        else:
            P.iloc[:, i] = clf.predict_proba(x_test)[:, 1]
        cols.append(name)
    P.columns=cols
    print('Done.\n')
    return P


def score_models(P, y_test, reg=None):
    # 评估各模型的效果
    model_report = DF(columns=['models'])
    for n, m in enumerate(P.columns):
        if reg:
            mse = metrics.mean_squared_error(y_test, P.loc[:, m])
            rmse = np.sqrt(mse)
            r2_score = metrics.r2_score(y_test, P.loc[:, m])
            mae = metrics.mean_absolute_error(y_test, P.loc[:, m])
            model_report.loc[n, 'models'] = m
            model_report.loc[n, 'mse'] = mse
            model_report.loc[n, 'rmse'] = rmse
            model_report.loc[n, 'r2_score'] = r2_score
            model_report.loc[n, 'mae'] = mae
        else:
            roc_auc = metrics.roc_auc_score(y_test, P.loc[:, m])
            # f1_score=metrics.f1_score(y_test,P.loc[:,m])
            model_report.loc[n, 'models'] = m
            model_report.loc[n, 'roc_auc_score'] = roc_auc
            # model_report.loc[n,'f1_score']=f1_score
    return model_report
