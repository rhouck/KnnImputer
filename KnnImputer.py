import random
import copy

import numpy as np
from scipy.spatial import KDTree


class KNNImputer(object):
    """Imputes missing data in numpy array using K Nearest Neighbors

    Attributes:
        k (int): Number neighbors used for imputing values.

    """
    
    def __init__(self, k):
        """
        Args:
            k (int): Number neighbors used for imputing values.

        """
        self.k = k
    
    def has_vals(self, x):
        f = lambda x: True if not np.isnan(x) else False
        return np.vectorize(f)(x)

    def bool_to_ind(self, row):
        return tuple([ind for ind, i in enumerate(row) if i])

    def get_column_combos(self, X):
        combos = set([tuple(i) for i in self.has_vals(X)])

        all_equal = lambda x: all(map(lambda i: i == x[0], x))
        combos = [i for i in combos if not all_equal(i)]

        return map(self.bool_to_ind, combos)
    
    def get_kdtree_w_inds(self, X):
        any_nans = np.isnan(X).any(axis=1)
        valid_inds = [ind for ind, i in enumerate(any_nans) if not i]
        return KDTree(X[valid_inds]), valid_inds
    
    def fit(self, X, y=None):
        self.X_train = copy.deepcopy(X)
        return self
    
    def transform(self, X, y=None):
        combos = self.get_column_combos(X)
        self.trees = {c: self.get_kdtree_w_inds(self.X_train[:, c]) for c in combos}
        
        def convert_inds(knn_inds, true_inds):
            max_ind = len(true_inds)
            return [true_inds[i] for i in knn_inds if i < max_ind]
        
        prefer_first = lambda x: x[0] if not np.isnan(x[0]) else x[1]
        
        X_imp = []
        for row in X:
            cols = self.has_vals(row)
            
            if all(cols):
                X_imp.append(row)
            
            else:
                cols = self.bool_to_ind(cols)
                knn_inds = self.trees[cols][0].query(row[list(cols)], self.k)[1]
                knn_inds = convert_inds(knn_inds, self.trees[cols][1])
                knn_means = np.nanmean(self.X_train[knn_inds], axis=0)           
                row_imp = [prefer_first(i) for i in zip(row, knn_means)] 
                X_imp.append(row_imp)
                
        return np.array(X_imp)


if __name__ == "__main__":
    X = np.random.randn(1000, 3)
    f = lambda x: x if random.random() > .2 else np.nan
    X = np.vectorize(f)(X)

    imp = KNNImputer(5)
    imp.fit(X)
    print(imp.transform(X[~np.isnan(X).all(axis=1)][:10]))