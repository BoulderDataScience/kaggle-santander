import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

sys.path.append('../..')
from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

if __name__ == '__main__':

    pipeline = Pipeline([
        ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
        ('std', StandardScaler())
    ])

    df_train = pd.read_csv('../../data/train.csv')
    df_target = df_train['TARGET']
    df_train = df_train.drop(['TARGET', 'ID'], axis=1)

    pipeline = pipeline.fit(df_train)
    X_train = pipeline.transform(df_train)
    y_train = df_target

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=50)

    print '\nAUC=%f: ' % roc_auc_score(y_test, model.predict_proba(X_test))