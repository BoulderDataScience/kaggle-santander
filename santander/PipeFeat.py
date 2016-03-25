import numpy as np
import pandas as pd

from math import log10
from statsmodels.distributions.empirical_distribution import ECDF

from sklearn.base import BaseEstimator

class Featurizer(BaseEstimator):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.customer = X
        return self.get_features()

    def scale01(self, feature, exclude = [0]):
        not_too_basic = [not f in exclude for f in feature]
        trim_feature = feature[not_too_basic]
        ecdf = ECDF(trim_feature)
        qtile = ecdf(feature)
        return qtile

    def characterize(self, col, feature, min_uniq_len = 5):

        ### INTEGER VARIABLES
        if feature.dtype == 'int64':

            # check for few unique values
            len_uniq = len(feature.unique())
            
            if len_uniq < min_uniq_len:
                
                if len_uniq == 1:
                    # print "Dropped %s no variation" % col
                    pred = None
                elif len_uniq == 2:
                    feature -= min(feature)
                    feature /= max(feature)
                    pred = pd.DataFrame({ col: feature })
                else:
                    pred = pd.get_dummies(feature, prefix = col, drop_first = True)
            
            elif max(feature % 3) == 0:
                # lots of columns have uniques divisible by 3

                if max(feature) > 999:
                    # These are large round payments perhaps on a loan
                    # ['imp_aport_var33_hace3', 'imp_aport_var33_ult1', 'var21']
                    pred = pd.DataFrame({ col: self.scale01(feature) })
                else:
                    # these might be loan lifetimes like 3, 6, 12, 18, 24
                    # they all start with 'num_'
                    pred = pd.DataFrame({ col: self.scale01(feature) })

            elif col == 'var3':
                # unclear gaussian mixture ????
                # if v > 25: delinquent = True
                # when this is @2 should be considered as base case
                pred = pd.DataFrame({ col: self.scale01(feature, exclude = [0, 1, 2]) })

            elif col == 'var15':
                # probably age
                pred = pd.DataFrame({ col: self.scale01(feature) })

            elif col == 'num_var4':
                # [0, 1, 2, 3, 4, 5, 6, 7] ????
                pred = pd.get_dummies(feature, prefix = col, drop_first = True)

            elif col == 'var36':
                # [0, 1, 2, 3, 99] ????
                pred = pd.get_dummies(feature, prefix = col, drop_first = True)

            else:
                print "Unknown int column ----> %s" % col
                pred = pd.DataFrame({ col: self.scale01(feature) })


        ### FLOAT VARIABLES
        elif feature.dtype == 'float64':

            # does it have any negative values??
            neg = [int(f < 0) for f in feature]
            is_negative = pd.DataFrame({ col+"_neg": neg })
            
            # what propotion of transactions end in .00?

            if max(neg) == 0:
                # strictly positive
                if col == 'var38':
                    # might be customer lifetime value
                    # non-missing, important
                    pred = pd.DataFrame({ col: self.scale01(feature) })
                else:
                    pred = pd.DataFrame({ col: self.scale01(feature) })
            else:

                # 
                feature_scaled = pd.DataFrame({ col: self.scale01(feature) })
                pred = pd.concat([is_negative, feature_scaled], axis = 1)

                pred = pd.DataFrame({ col: self.scale01(feature) })

        else:
            print "Feature %s is neither int nor float" % col
            pred = feature
        
        return pred

    def get_features(self):

        frames = []
        nct = 0
        for col, feature in self.customer.iteritems():
            wide_feature = self.characterize(col, feature)
            if wide_feature is None:
                nct += 1
            else:
                wide_feature.reset_index(inplace=True, drop=True) # WOW INDEX JUST WOW
                frames.append(wide_feature)

        wide_customer = pd.concat(frames, axis = 1)
        
        print "Dropped %d features due to no variation" % nct
        print "Introduced %d features" % (wide_customer.shape[1]-(self.customer.shape[1]-nct))
        return wide_customer
