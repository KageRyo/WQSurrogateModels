"""Exercise composed reconstruction, reduced and masked inference paths."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from edge_optimization import FEATURES, forest, predict
from edge_onnx_missing_evaluate import convert_bundle
from edge_onnx_evaluate import session


@pytest.mark.parametrize("missing", [["BOD"], ["NH3N"], ["BOD", "NH3N"]])
@pytest.mark.parametrize(
    "mode", ["inference_dropout", "reduced_retraining", "indicator_reconstruction"]
)
def test_composed_missing_pipeline_ignores_unavailable_columns(tmp_path, missing, mode):
    rng = np.random.default_rng(34)
    frame = pd.DataFrame(rng.uniform(1, 20, (100, 5)), columns=FEATURES)
    available = [c for c in FEATURES if c not in missing]
    y = frame.DO * 2 + frame.EC / 4
    config = {"forest_random_state": 0, "training_jobs": 1}
    indicators = {c: forest(config, 3, frame[available], frame[c]) for c in missing}
    restored = frame.copy()
    for c, model in indicators.items():
        restored[c] = model.predict(frame[available])
    train = (
        frame[available]
        if mode == "reduced_retraining"
        else restored
        if mode == "indicator_reconstruction"
        else frame
    )
    bundle = dict(
        wqi_model=forest(config, 4, train, y),
        indicator_models=indicators,
        missing_indicators=missing,
        available_features=available,
    )
    graph = convert_bundle(bundle, mode, "0" * 64)
    path = tmp_path / "pipeline.onnx"
    path.write_bytes(graph.SerializeToString())
    runtime = session(path)
    got = runtime.run(["score"], {"features": frame.to_numpy(dtype=np.float32)})[
        0
    ].reshape(-1)
    np.testing.assert_allclose(got, predict(bundle, frame, mode), atol=1e-5, rtol=1e-6)
    poisoned = frame.copy()
    poisoned[missing] = np.nan
    np.testing.assert_array_equal(
        got,
        runtime.run(["score"], {"features": poisoned.to_numpy(dtype=np.float32)})[
            0
        ].reshape(-1),
    )
    if mode == "indicator_reconstruction":
        actual = runtime.run(
            ["model_features"], {"features": frame.to_numpy(dtype=np.float32)}
        )[0]
        np.testing.assert_allclose(actual, restored.to_numpy(), atol=1e-5, rtol=1e-6)


def test_nano_worker_rejects_mislabeled_graph_and_feature_order():
    from edge_onnx_missing_benchmark import validate_contract

    contract = dict(
        mode="indicator_reconstruction",
        missing=["BOD"],
        input_features=list(FEATURES),
        available_features=[c for c in FEATURES if c != "BOD"],
    )
    validate_contract(contract, "indicator_reconstruction", ["BOD"], list(FEATURES))
    with pytest.raises(ValueError, match="metadata"):
        validate_contract(contract, "inference_dropout", ["BOD"], list(FEATURES))
    with pytest.raises(ValueError, match="metadata"):
        validate_contract(
            contract, "indicator_reconstruction", ["NH3N"], list(FEATURES)
        )
    with pytest.raises(ValueError, match="feature order"):
        validate_contract(
            contract, "indicator_reconstruction", ["BOD"], list(reversed(FEATURES))
        )
