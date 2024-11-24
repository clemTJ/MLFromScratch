import numpy as np

from base import Model
from decision_tree import DecisionTree


class RandomForest(Model):

  def __init__(self, nb_trees : int = 10, max_depth: int = 10, min_sample_split: int = 2, n_features : int = None):
    super().__init__()
    self.nb_trees = nb_trees
    self.max_depth = max_depth
    self.min_sample_split = min_sample_split
    self.n_features = n_features
    self.trees = list()

  def fit(self, X, y):
    n_samples = X.shape[0]

    for _ in range(self.nb_trees):
      tree = DecisionTree(max_depth=self.max_depth, min_sample_split=self.min_sample_split, n_features=self.n_features)

      idx_samples = np.random.choice(n_samples, n_samples, replace=True)
      tree.fit(X[idx_samples], y[idx_samples])

      self.trees.append(tree)

  def predict(self, X):
    predictions = np.array([tree.predict(X) for tree in self.trees])
    predictions = np.swapaxes(predictions, 0, 1)
    predictions = np.array([self._most_common(pred) for pred in predictions])
    return predictions
