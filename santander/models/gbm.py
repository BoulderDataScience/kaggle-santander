import sys
sys.path.append('../..')

import pandas as pd

from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

if __name__ == '__main__':
    pipeline = Pipeline([
        ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS + CORRELATED_COLUMNS)),
    ])

    df_train = pd.read_csv('../../data/train.csv')
    df_target = df_train['TARGET']
    df_train = df_train.drop(['TARGET', 'ID'], axis=1)

    pipeline = pipeline.fit(df_train)
    X_train = pipeline.transform(df_train)
    y_train = df_target

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    # hand-tuned params
    # learning_rate = 0.025
    # n_estimators = 400
    # min_child_weight = 5
    # max_depth = 4

    # better params using columns column subsampling, longer training
    # learning_rate = 0.02
    # n_estimators = 500
    # max_depth = 6
    # subsample = 1
    # colsample_bytree = 0.85
    # min_child_weight = 1  # default

    # best params so far using column/row subsampling, even longer training
    learning_rate = 0.01
    n_estimators = 800
    max_depth = 6
    subsample = 0.9
    colsample_bytree = 0.85
    min_child_weight = 1  # default

    gbm = XGBClassifier(seed=0, learning_rate=learning_rate, n_estimators=n_estimators,
                        min_child_weight=min_child_weight, max_depth=max_depth,
                        colsample_bytree=colsample_bytree, subsample=subsample)
    gbm = gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=['logloss', 'auc'])

    print '\nAUC=%f: ' % roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])
