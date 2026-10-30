from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from wqsurrogatemodels.wqi import direct_wqi5_score


class DirectWQI5(RegressorMixin, BaseEstimator):
    """Serializable adapter for the artifact-free complete-input baseline."""

    def fit(self, frame, y=None):
        return self

    def predict(self, frame):
        return np.asarray([
            direct_wqi5_score(do=row.DO, bod=row.BOD, nh3n=row.NH3N, ec=row.EC, ss=row.SS)
            for row in frame.itertuples(index=False)
        ])
