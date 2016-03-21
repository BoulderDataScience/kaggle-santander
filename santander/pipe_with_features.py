import sys
import os
sys.path.append(os.path.abspath('..'))


import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, plot_importance

from santander.utils import ColumnDropper
from santander.utils import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

from santander.PipeFeat import Featurizer


df_train = pd.read_csv('../data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)


# X_train = df_train.copy()
# y_train = df_target
# 
# X_train, X_test, y_train, y_test= train_test_split(X_train, y_train, test_size=0.3, random_state=0)
# xgb = XGBClassifier(seed=0)
# xgb = xgb.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_test, y_test)])

# plot_importance(xgb)


pipeline = Pipeline([
        ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
        ('feat', Featurizer()) #,
#        ('std', StandardScaler()),
#        ('pca', PCA(n_components=150))
    ])

pipeline = pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
y_train = df_target

X_train, X_test, y_train, y_test= train_test_split(X_train, y_train, test_size=0.3, random_state=0)
xgb = XGBClassifier(seed=0)
xgb = xgb.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_test, y_test)])

# plot_importance(xgb)

