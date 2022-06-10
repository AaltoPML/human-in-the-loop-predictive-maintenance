import numpy as onp
import collections

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class Human:
    def __init__(self, experience, depth):
        self._experience = experience
        self._depth = depth
        self._tree = None
        self._X = None
        self._y = None
        self._accuracy = 0.0

    def fit(self, X, y):
        cnt = collections.Counter(y)
        self._tree = DecisionTreeClassifier(
            random_state=42, max_depth=self._depth,
            criterion="entropy", class_weight={0: cnt[1], 1: cnt[0]})
        num_train = int(len(X) * self._experience)
        idx = onp.random.choice(onp.arange(len(X)), num_train, replace=False)
        self._X, self._y = X[idx], y[idx]
        self._tree.fit(self._X, self._y)
        self._accuracy = accuracy_score(
            self._tree.predict(onp.array(self._X)), self._y)
        return self._X, self._y

    def refit(self, new_X, new_y):
        X_all = onp.concatenate((self._X, new_X), axis=0)
        y_all = onp.concatenate((self._y, new_y), axis=0)

        self._tree.fit(X_all, y_all)

    def predict(self, X):
        return self._tree.predict(X)

    def __copy__(self):
        return Human(experience=self._experience,
                     depth=self._depth)
