import numpy as np
import time

from typing import Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin

from .bagging import bootstrap_sample


class RandomForestClassifierCustom(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            base_estimator,
            n_estimators: int = 10,
            n_features: Optional[int] = None,
            max_samples: Optional[Union[int, float]] = 1.0,
            min_samples: Optional[Union[int, float]] = 0.0,
            random_state: Optional[int] = None,
            bootstrap: bool = True
    ):
        self.base_estimator_ = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.n_features = n_features

        self.estimators_ = []
        self.estimators_features_ = []

    def fit(self, X: np.array, y: np.array) -> 'RandomForestClassifierCustom':
        self.estimators_ = []

        seed = self.random_state if self.random_state is not None else int(time.time() % 1000)
        n_samples, _ = X.shape

        if self.n_features is None:
            n_features = round(np.sqrt(X.shape[0]))
        else:
            n_features = self.n_features

        for i in range(self.n_estimators):
            np.random.seed(seed + i)

            indices = np.arange(n_samples)
            selected = bootstrap_sample(indices, self.min_samples, self.max_samples, self.bootstrap)
            features = np.random.choice(np.arange(X.shape[1]), size=n_features, replace=False)

            X_train = X[selected][:, features]
            y_train = y[selected]

            estimator = self.base_estimator_.__class__(**self.base_estimator_.get_params())
            estimator.fit(X_train, y_train)

            self.estimators_.append(estimator)
            self.estimators_features_.append(features)

        return self

    def predict_proba(self, X: np.array) -> np.array:
        predictions = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            estimator = self.estimators_[i]
            features = self.estimators_features_[i]
            predictions += estimator.predict(X[:, features])
        return predictions / self.n_estimators

    def predict(self, X):
        return np.int64(np.round(self.predict_proba(X)))
