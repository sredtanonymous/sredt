"""Implementation of the SR-Enhanced CART algorithm."""
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from copy import deepcopy
import pandas as pd

class Node:
    def __init__(self, predicted_class,class_stats):
        self.predicted_class = predicted_class
        self.class_stats = class_stats
        self.feature_eq = None
        self.threshold = None
        self.left = None
        self.right = None

class SRClassifier:
    def __init__(self, max_depth=None, random_state=0):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.n_classes_ = len(set(list(np.array(y).flatten())))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _best_split(self, X, y):
        if len(X)<=0:
            return None, None
        def _gini_formula(y, y_pred, w, mode):
            """Calculate the gini."""
            #print(y,y_pred)
            m = y.size
            if m <= 1:
                return 1
            num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
            best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
            best_idx, best_thr = None, None
            thresholds, classes = zip(*sorted(zip(y_pred, y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
            if mode=="threshold":
                if not best_thr:
                    return None
                return best_thr
            elif mode=="gini":
                return best_gini
        _gini = lambda y, y_pred, w: _gini_formula(y, y_pred, w, "gini")
        _gini_threshold = lambda y, y_pred, w: _gini_formula(y, y_pred, w, "threshold")
        gini = make_fitness(_gini, greater_is_better=False)
        est = SymbolicRegressor(parsimony_coefficient=.001,generations=10,n_jobs=1,low_memory=True,population_size=400,tournament_size=200,
                     function_set=('add','sub','mul','div'),random_state=self.random_state,const_range=(-1,1),verbose=1,metric=gini)
        est.fit(X, y)
        best_thr = _gini_threshold(y, est.predict(X), None)
        if not best_thr:
            return None, None
        else:
            return deepcopy(est), _gini_threshold(y, est.predict(X), None)

    def _grow_tree(self, X, y, depth=0):
        class_stats = [f"Class{i}: {np.sum(y == i)}" for i in range(self.n_classes_)]
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class, class_stats=class_stats)
        if depth < self.max_depth and max(num_samples_per_class)<len(y)*0.95:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = idx.predict(X) < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_eq = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if node.feature_eq.predict([inputs]) < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
