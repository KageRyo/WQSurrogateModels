"""Real converter execution for each newly evaluated non-RF model family."""

from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_onnx_family_evaluate import register_external
from edge_onnx_evaluate import session


@pytest.mark.parametrize("family", ["lr", "mpr", "svm", "lightgbm", "xgboost"])
def test_registered_family_full_pipeline_executes(tmp_path, family):
    register_external()
    rng = np.random.default_rng(19)
    X = rng.uniform(1, 10, (150, 5)).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1]
    models = {
        "lr": LinearRegression(),
        "mpr": LinearRegression(),
        "svm": SVR(C=10),
        "lightgbm": LGBMRegressor(n_estimators=3, n_jobs=1, verbosity=-1),
        "xgboost": XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1, device="cpu"),
    }
    steps = [("imputer", SimpleImputer())]
    if family == "mpr":
        steps.append(("poly", PolynomialFeatures(include_bias=False)))
    if family in ["lr", "mpr", "svm"]:
        steps.append(("scaler", StandardScaler()))
    pipeline = Pipeline([*steps, ("model", models[family])]).fit(X, y)
    graph = convert_sklearn(
        pipeline,
        initial_types=[("features", FloatTensorType([None, 5]))],
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    path = tmp_path / "pipeline.onnx"
    path.write_bytes(graph.SerializeToString())
    probe = X[:20].copy()
    probe[0, 2] = np.nan
    output = session(path).run(None, {"features": probe})[0].reshape(-1)
    np.testing.assert_allclose(output, pipeline.predict(probe), rtol=1e-5, atol=1e-4)


def test_checksum_guard_survives_python_optimized_mode(tmp_path):
    import json
    import subprocess

    root = tmp_path / "source"
    artifact = root / "models" / "fixture.joblib"
    artifact.parent.mkdir(parents=True)
    # This is deliberately not a pickle. Bypassing the guard reaches a different
    # deserialization error, so the diagnostic proves rejection before loading.
    artifact.write_bytes(b"not a serialized model")
    (root / "models/production_model_manifest.json").write_text(
        json.dumps(
            dict(
                artifacts=[
                    dict(
                        model_type="lr",
                        production_artifact="models/fixture.joblib",
                        sha256="0" * 64,
                    )
                ]
            )
        )
    )
    output = tmp_path / "results" / "rejected"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/edge_onnx_family_evaluate.py"
            ),
            "--data-root",
            str(root),
            "--model",
            "lr",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Production artifact checksum mismatch" in result.stderr
    assert not output.exists()
