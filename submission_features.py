import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from santander.preprocessing import ColumnDropper
from santander.preprocessing import CORRELATED_COLUMNS
from santander.feature_extraction import BOW, Featurizer

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=CORRELATED_COLUMNS)),
    ('feat', Featurizer())
])

df_train = pd.read_csv('data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)

df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)

# combine for a second to run bow/featurizer
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
num_zero = (df_all == 0).sum(axis=1)
bow = BOW().fit(df_all).transform(df_all)
df_all = pd.concat([df_all, bow], axis=1)
df_all['num_zero'] = num_zero
df_all = pipeline.fit(df_all).transform(df_all)

X_train = df_all.iloc[:df_train.shape[0], :]
X_test = df_all.iloc[df_train.shape[0]:, :]
y_train = df_target
ID_test = df_id

# best params so far using column/row subsampling, even longer training
learning_rate = 0.01
n_estimators = 800
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 1  # default

xgb = XGBClassifier(seed=0, learning_rate=learning_rate, n_estimators=n_estimators,
                    min_child_weight=min_child_weight, max_depth=max_depth,
                    colsample_bytree=colsample_bytree, subsample=subsample)
xgb = xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='auc')

importances = xgb.booster().get_fscore()
df_importance = pd.DataFrame(zip(importances.keys(), importances.values()), columns=['feature', 'importance'])
print df_importance.sort_values('importance', ascending=False).reset_index(drop=True)

y_pred = xgb.predict_proba(X_test)
submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, 1]})
submission.to_csv('submission_features.csv', index=False)
print 'Wrote submission_features.csv'
