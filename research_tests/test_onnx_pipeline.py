"""Mandatory in the isolated research CI; missing conversion packages are errors."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_optimization import FEATURES, forest
from edge_onnx_evaluate import compare, convert_pipeline, session


def test_fitted_imputer_and_tree_forward_path(tmp_path):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.uniform(1, 30, size=(80, len(FEATURES))), columns=FEATURES)
    target = frame.DO.to_numpy() * 2 + frame.BOD.to_numpy()
    frame.loc[0, "BOD"] = np.nan
    fitted = forest({"forest_random_state": 0, "training_jobs": 1}, 5, frame, target)
    path = tmp_path / "model.onnx"
    path.write_bytes(convert_pipeline(fitted).SerializeToString())
    X = frame.astype(np.float32)
    output = session(path).run(None, {"features": X.to_numpy()})[0].reshape(-1)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, fitted.predict(X), atol=2e-5, rtol=1e-6)
    report = compare(np.array([50.0]), np.array([50.0]), np.array([50.00001]))
    assert report["native64_to_onnx32"]["above_tolerance"] == 1
