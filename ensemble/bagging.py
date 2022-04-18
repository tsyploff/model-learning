import numpy as np
import time

from typing import Callable, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin


def out_of_bag(array, bag):
    return array[np.logical_not(np.any(array.reshape(-1, 1) == bag, axis=1))]


def out_of_bag_score(
        estimator,
        scoring: Optional[Callable[[np.array, np.array], float]],
        X: np.array,
        y: np.array,
        indices: np.array,
        selected: np.array
):
    oob = out_of_bag(indices, selected)
    if scoring is None:
        return estimator.score(X[oob], y[oob])
    else:
        return scoring(estimator.predict(X[oob]), y[oob])


class BaggingClassifierCustom(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            base_estimator,
            n_estimators: int = 10,
            max_samples: Optional[Union[int, float]] = 1.0,
            min_features: Optional[Union[int, float]] = 0.0,
            oob_score: bool = False,
            oob_scoring: Optional[Callable[[np.array, np.array], float]] = None,
            random_state: Optional[int] = None,
            bootstrap: bool = True
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.min_features = min_features
        self.oob_score = oob_score
        self.oob_scoring = oob_scoring
        self.random_state = random_state
        self.bootstrap = bootstrap

        self.estimators_ = []
        self.oob_score_ = None

    def fit(self, X: np.array, y: np.array) -> 'BaggingClassifierCustom':
        self.estimators_ = []
        self.oob_score_ = None

        seed = self.random_state if self.random_state is not None else int(time.time() % 1000)
        n_samples, n_features = X.shape

        oob_score = 0

        for i in range(self.n_estimators):
            np.random.seed(seed + i)

            indices = np.arange(n_samples)
            selected = np.random.choice(
                indices,
                size=np.random.randint(1, self.max_samples),
                replace=self.bootstrap
            )

            X_train = X[selected]
            y_train = y[selected]

            estimator = self.base_estimator.__class__()
            estimator.set_params(**self.base_estimator.get_params())
            estimator.fit(X_train, y_train)

            self.estimators_.append(estimator)

            if self.oob_score:
                oob_score += out_of_bag_score(estimator, self.oob_scoring, X, y, indices, selected)

        if self.oob_score:
            self.oob_score_ = oob_score / self.n_estimators

        return self

    def predict(self, X: np.array) -> np.array:
        pass
