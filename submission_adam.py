import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from santander.preprocessing import ColumnDropper
from santander.preprocessing import CORRELATED_COLUMNS
from santander.feature_extraction import Featurizer, BOW

# read in both sets of data
train = pd.read_csv('data/train.csv')
train['train_or_test'] = pd.Series('train', index=train.index)
test = pd.read_csv('data/test.csv')
test['train_or_test'] = pd.Series('test', index=test.index)

# grab the answers then drop
y_train = train['TARGET']
train = train.drop('TARGET',1)

# save ID for later
ID_test = test['ID']

# concat and drop ID, reset index
all_obs = pd.concat([train, test], axis = 0)
all_obs = all_obs.drop('ID', 1)
all_obs.reset_index(drop=True, inplace=True)

# combine spanish bag of words
spanish = BOW().fit(all_obs).transform(all_obs)
all_obs = pd.concat([all_obs, spanish], 1)

# initialize pipeline
pipeline = Pipeline([
        ('cd', ColumnDropper(drop=CORRELATED_COLUMNS)),
        ('feat', Featurizer())
    ])

# run all_obs through featurization
pipeline = pipeline.fit(all_obs)
all_obs_featz = pipeline.transform(all_obs)

# grab train/test data back out
X_train = all_obs_featz[all_obs_featz['train_or_test']=='train']
X_train = X_train.drop('train_or_test',1)
X_test = all_obs_featz[all_obs_featz['train_or_test']=='test']
X_test = X_test.drop('train_or_test',1)


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

y_pred = xgb.predict_proba(X_test)
submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, 1]})
submission.to_csv('submission_adam.csv', index=False)
print 'Wrote submission_adam.csv'