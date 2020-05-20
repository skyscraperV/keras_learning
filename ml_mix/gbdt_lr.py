import numpy as np
import random
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(10)

X, Y = make_classification(n_samples=1000, n_features=30)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=7, test_size=0.2)


def gbdt_lr_mix():
    gbdtModel = GradientBoostingClassifier(n_estimators=10)
    gbdtModel.fit(X_train, Y_train)
    oneHot = OneHotEncoder()
    train_leafs_inds = gbdtModel.apply(X_train)[:, :, 0]
    print(np.shape(train_leafs_inds))
    print(gbdtModel.estimators_)
    oneHot.fit(train_leafs_inds)
    lrModel = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    lrModel.fit(oneHot.transform(train_leafs_inds), Y_train)
    Y_pred = lrModel.predict_proba(oneHot.transform(gbdtModel.apply(X_test)[:, :, 0]))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('gbdt + lr: ', auc)
    return fpr, tpr

def pure_lr():
    lrMpdel = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    lrMpdel.fit(X_train, Y_train)
    Y_pred = lrMpdel.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('lr: ', auc)
    return fpr, tpr

def pure_xgboost():
    xgbModel = xgb.XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=100, colsample_bytree=0.5)
    xgbModel.fit(X_train, Y_train)
    Y_pred = xgbModel.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('xgboost: ', auc)
    return fpr, tpr


def xgboost_lr():
    xgbModel = xgb.XGBClassifier(nthread=4, learning_rate=0.08, n_estimators=100, colsample_bytree=0.5)
    xgbModel.fit(X_train, Y_train)
    oneHot = OneHotEncoder()
    oneHot.fit(xgbModel.apply(X_train))
    lrModel = LogisticRegression(n_jobs=4, C=0.1, penalty='l2')
    lrModel.fit(oneHot.transform(xgbModel.apply(X_train)), Y_train)
    Y_pred = lrModel.predict_proba(oneHot.transform(xgbModel.apply(X_test)))[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    print('xgb + lr: ', auc)
    return fpr, tpr


if __name__ == '__main__':
    fpr_lr, tpr_lr = pure_lr()
    fpr_gbdt_lr, tpr_gbdt_lr = gbdt_lr_mix()
    fpr_xgb, tpr_xgb = pure_xgboost()
    fpr_xgb_lr, tpr_xgb_lr = xgboost_lr()


    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBT + LR')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_xgb, tpr_xgb, label='XGB')
    plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_gbdt_lr, tpr_gbdt_lr, label='GBT + LR')
    plt.plot(fpr_lr, tpr_lr, label='LR')
    plt.plot(fpr_xgb, tpr_xgb, label='XGB')
    plt.plot(fpr_xgb_lr, tpr_xgb_lr, label='XGB + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()

