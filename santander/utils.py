import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler


def pd_to_vw(filename, features, labels, reweight_importance=False, namespace_to_features=None):
    """
    Write pandas features/labels to vw file format

    :param filename: vw output filename
    :param features: pandas dataframe of features (n x p)
    :param labels: labels (series or array, n x 1)
    :param reweight_importance: whether to explicitly reweight importances to deal with class imbalance or leave at 1.0
    :param namespace_to_features: namespace to feature mapping in format
    {'namespace1': [feature1 feature2 ...], 'namespace2': [feature1 feature2 ...], ...}. By default, all features are
    put in a single namespace 'a' for easy quadratic/cubic featurizing
    :return:
    """
    n = features.shape[0]

    if features.shape[0] != labels.shape[-1]:
        raise ValueError('Features and labels must have same length')

    # reweight importance to deal with class imbalance
    if reweight_importance:
        counts = np.bincount(labels)
        bins = np.nonzero(counts)[0]
        importance = deepcopy(np.asarray(labels))
        for b, c in zip(bins, counts):
            importance[importance == b] = n/float(c)
    else:
        importance = np.ones(labels.shape[0])

    # by default, all variables go in one namespace 'a'
    if namespace_to_features is None:
        namespace_to_features = {'a': list(features.columns)}

    # labels must be -1/1 for log loss
    labels = MinMaxScaler((-1, 1)).fit_transform(labels)

    with open(filename, 'w') as f:
        for i in range(n):
            line = '%d ' % labels[i]
            line += '%f ' % importance
            for namespace in namespace_to_features:
                line += '|%s ' % namespace
                d = features.loc[i, namespace_to_features[namespace]].to_dict()
                line += ' '.join(['%s:%f' % (k, v) for k, v in d.iteritems()])
            f.write(line+'\n')
            if i % int(n/10.0) == 0:
                print '%d/%d' % (i, n)

    print 'Wrote %s' % filename
