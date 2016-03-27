from sklearn.base import BaseEstimator


ZERO_VARIANCE_COLUMNS = [
    'ind_var2_0', 'ind_var2', 'ind_var27_0', 'ind_var28_0', 'ind_var28', 'ind_var27', 'ind_var41', 'ind_var46_0',
    'ind_var46', 'num_var27_0', 'num_var28_0', 'num_var28', 'num_var27', 'num_var41', 'num_var46_0', 'num_var46',
    'saldo_var28', 'saldo_var27', 'saldo_var41', 'saldo_var46', 'imp_amort_var18_hace3', 'imp_amort_var34_hace3',
    'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3', 'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3',
    'num_var2_0_ult1', 'num_var2_ult1', 'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
    'num_trasp_var33_out_hace3', 'saldo_var2_ult1', 'saldo_medio_var13_medio_hace3'
]

CORRELATED_COLUMNS = [
    'ind_var29_0', 'ind_var29', 'num_var6', 'num_var29', 'ind_var13_medio', 'num_var13_medio_0', 'num_var13_medio',
    'num_meses_var13_medio_ult3', 'ind_var18', 'num_var18_0', 'num_var18', 'num_var20_0', 'num_var20', 'ind_var26',
    'ind_var25', 'ind_var32', 'ind_var34', 'ind_var37', 'ind_var39', 'num_var29_0', 'delta_imp_amort_var18_1y3',
    'num_var26', 'num_var25', 'num_var32', 'num_var34', 'delta_imp_amort_var34_1y3', 'num_var37', 'num_var39',
    'saldo_var29', 'saldo_medio_var13_medio_ult1', 'delta_num_aport_var13_1y3', 'delta_num_aport_var17_1y3',
    'delta_num_aport_var33_1y3', 'delta_num_reemb_var13_1y3', 'num_reemb_var13_ult1', 'delta_num_reemb_var17_1y3',
    'delta_num_reemb_var33_1y3', 'num_reemb_var33_ult1', 'delta_num_trasp_var17_in_1y3',
    'delta_num_trasp_var17_out_1y3', 'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3',
    'num_trasp_var33_out_ult1', 'delta_num_venta_var44_1y3'
]


class ColumnDropper(BaseEstimator):
    """
    Drop columns based on name
    """
    def __init__(self, drop=[]):
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.drop, axis=1)


class ColumnDropperVarianceThreshold(ColumnDropper):
    """
    Drop columns if variance <= threshold
    """
    def __init__(self, variance_threshold):
        self.variance_threshold = variance_threshold
        super(ColumnDropperVarianceThreshold, self).__init__()

    def fit(self, X, y=None):
        for col in X.columns:
            if X[col].var() <= self.variance_threshold:
                self.drop.append(col)


class ColumnDropperCorrelationThreshold(ColumnDropper):
    """
    Drop columns if correlation >= threshold
    """
    def __init__(self, correlation_threshold):
        self.correlation_threshold = correlation_threshold
        super(ColumnDropperCorrelationThreshold, self).__init__()

    def fit(self, X, y=None):
        corr = X.corr().abs()
        for i in range(corr.shape[0]-1):
            for j in range(i+1, corr.shape[1]):
                if corr.values[i, j] >= self.correlation_threshold and corr.columns[j] not in self.drop:
                    self.drop.append(corr.columns[j])
