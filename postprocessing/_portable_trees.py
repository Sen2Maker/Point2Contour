"""Inference-only numeric trees, independent of sklearn's pickle internals."""
import numpy as np
from scipy.special import expit


class FrozenTrees:
    def __init__(self, archive, key):
        self.baseline = float(archive[key + '_baseline'])
        self.n_features_in_ = int(archive[key + '_features'])
        self.classes_ = archive[key + '_classes'].copy()
        self.trees = [archive[key + '_tree_' + str(i)].copy()
                      for i in range(int(archive[key + '_count']))]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('Invalid editor feature matrix')
        result = np.full(len(X), self.baseline, dtype=np.float64)
        for tree in self.trees:
            indices = np.zeros(len(X), dtype=np.int64)
            active = np.arange(len(X))
            while len(active):
                nodes = tree[indices[active]]
                leaf = nodes['is_leaf'].astype(bool)
                result[active[leaf]] += nodes['value'][leaf]
                active = active[~leaf]
                nodes = nodes[~leaf]
                values = X[active, nodes['feature_idx']]
                left = np.where(np.isnan(values), nodes['missing_go_to_left'],
                                values <= nodes['threshold'])
                indices[active] = np.where(left, nodes['left'], nodes['right'])
        return result

    def predict_proba(self, X):
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError('Only the frozen binary editors are supported')
        positive = expit(self.predict(X))
        return np.column_stack((1.0 - positive, positive))


def load_editors(path):
    with np.load(path, allow_pickle=False) as archive:
        return {str(key): FrozenTrees(archive, str(key)) for key in archive['keys']}
