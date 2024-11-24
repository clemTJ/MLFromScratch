import numpy as np
from collections import Counter


from base import Model


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree(Model):

    def __init__(self, max_depth: int = 10, min_sample_split: int = 2, n_features: int = None):
        super().__init__()
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_features = n_features if self.n_features is None else min(self.n_features, n_features)

        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth: int = 0):
        n_samples, n_features = X.shape

        if len(np.unique(y)) == 1 or n_samples < self.min_sample_split or depth > self.max_depth:
            value = self._most_common(y)
            return Node(value=value)

        features = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self._best_choice(X, y, features)

        left_idx = np.where(X[:, best_feature] <= best_threshold)
        right_idx = np.where(X[:, best_feature] > best_threshold)

        left_node = self._grow_tree(X[left_idx], y[left_idx], depth=depth + 1)
        right_node = self._grow_tree(X[right_idx], y[right_idx], depth=depth + 1)

        return Node(best_feature, best_threshold, left=left_node, right=right_node)

    def _best_choice(self, X, y, features):
        max_ig = 0
        feature, threshold = None, None

        for ftr in features:
            x_feature = X[:, ftr]

            for value in np.unique(x_feature):

                information_gain = self._information_gain(X, y, ftr, value)
                if information_gain > max_ig:
                    max_ig = information_gain
                    feature = ftr
                    threshold = value

        return feature, threshold

    def _information_gain(self, X, y, feature, threshold):
        n_samples = X.shape[0]
        parent_entropy = self._entropy(y)

        left_idx = np.where(X[:, feature] <= threshold)[0]
        left_entropy = self._entropy(y[left_idx]) * len(left_idx) / n_samples

        right_idx = np.where(X[:, feature] > threshold)[0]
        right_entropy = self._entropy(y[right_idx]) * len(right_idx) / n_samples

        information_gain = parent_entropy - left_entropy - right_entropy
        return information_gain

    def _entropy(self, values: list):
        values = np.divide(np.bincount(values), len(values))
        return -np.sum(np.array([val * np.log(val) for val in values if val > 0]))

    def _most_common(self, values: list):
        counter = Counter(values)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return [self._traverse_tree(x) for x in X]

    def _traverse_tree(self, x, node: Node = None):
        if node is None:
            node = self.root

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
