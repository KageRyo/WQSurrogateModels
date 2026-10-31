from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import joblib
import pytest

from wqsurrogatemodels import __version__, api
from wqsurrogatemodels.enums import ModelTypeEnum
from wqsurrogatemodels.services import WaterQualityService
from wqsurrogatemodels.settings import Settings


FIXTURES = Path(__file__).parent / "fixtures" / "api_v2"
ASSESSMENT_FIELDS = {
    "score",
    "category",
    "rating_range",
    "model_type",
    "latency_ms",
    "assessment",
    "warnings",
}


class ConstantSurrogate:
    """A safe, deterministic model used only by the integration test fixture."""

    def predict(self, frame):  # type: ignore[no-untyped-def]
        return [73.25] * len(frame)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def read_json_fixture(name: str) -> dict[str, float | str]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def read_csv_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def assert_assessment_contract(payload: dict) -> None:
    assert set(payload) == ASSESSMENT_FIELDS
    assert isinstance(payload["score"], float)
    assert 0 <= payload["score"] <= 100
    assert payload["category"] in {"Excellent", "Good", "Fair", "Poor", "Bad", "Terrible"}
    assert isinstance(payload["rating_range"], str)
    assert payload["model_type"] in {model.value for model in ModelTypeEnum}
    assert isinstance(payload["latency_ms"], float)
    assert set(payload["assessment"]) == {"DO", "BOD", "NH3N", "EC", "SS"}
    assert isinstance(payload["warnings"], list)


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


@pytest.mark.anyio
async def test_v2_health_and_ready_contract(client: httpx.AsyncClient) -> None:
    health = await client.get("/api/v2/health")
    ready = await client.get("/api/v2/ready")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "message": "WQSurrogateModels v2 is healthy.",
        "default_model": "direct_wqi5",
        "version": __version__,
    }
    assert ready.status_code == 200
    ready_payload = ready.json()
    assert set(ready_payload) == {
        "status",
        "message",
        "default_model",
        "dataset_available",
        "dataset_required",
        "models",
    }
    assert ready_payload["status"] == "ready"
    assert ready_payload["message"] == "WQSurrogateModels v2 is ready to serve assessments."
    assert ready_payload["default_model"] == "direct_wqi5"
    assert ready_payload["dataset_required"] is False
    assert ready_payload["models"][0] == {"model_type": "direct_wqi5", "available": True}


@pytest.mark.anyio
async def test_v2_models_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v2/models")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"models", "default_model"}
    assert payload["default_model"] == "direct_wqi5"
    direct_model = next(model for model in payload["models"] if model["model_type"] == "direct_wqi5")
    assert direct_model == {
        "model_type": "direct_wqi5",
        "available": True,
        "version": "direct-wqi5-v1",
    }
    assert all(set(model) == {"model_type", "available", "version"} for model in payload["models"])
    assert "/" not in str(payload["models"])


@pytest.mark.anyio
async def test_v2_single_assessment_contract(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v2/assessment", json=read_json_fixture("single_assessment.json"))

    assert response.status_code == 200
    payload = response.json()
    assert_assessment_contract(payload)
    assert payload["score"] == 89.542
    assert payload["model_type"] == "direct_wqi5"
    assert payload["warnings"] == []


@pytest.mark.anyio
async def test_v2_single_assessment_returns_range_warnings(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v2/assessment", json=read_json_fixture("out_of_range_assessment.json"))

    assert response.status_code == 200
    payload = response.json()
    assert_assessment_contract(payload)
    assert payload["warnings"] == [
        "DO=151.0 is outside the recommended range [0, 150].",
        "BOD=201.0 is outside the recommended range [0, 200].",
        "NH3N=51.0 is outside the recommended range [0, 50].",
        "EC=50001.0 is outside the recommended range [0, 50000].",
        "SS=5001.0 is outside the recommended range [0, 5000].",
    ]


@pytest.mark.anyio
async def test_v2_csv_summary_contract(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v2/assessment/csv/summary",
        data={"model_type": "direct_wqi5"},
        files={"file": ("valid_assessments.csv", read_csv_fixture("valid_assessments.csv"), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert_assessment_contract(payload)
    assert payload["model_type"] == "direct_wqi5"
    assert payload["score"] == 84.142


@pytest.mark.anyio
async def test_v2_csv_rows_contract(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v2/assessment/csv/rows",
        data={"model_type": "direct_wqi5"},
        files={"file": ("valid_assessments.csv", read_csv_fixture("valid_assessments.csv"), "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"scores", "model_type", "latency_ms"}
    assert payload["scores"] == [89.542, 78.741]
    assert payload["model_type"] == "direct_wqi5"
    assert isinstance(payload["latency_ms"], float)


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", ["/api/v2/assessment/csv/summary", "/api/v2/assessment/csv/rows"])
async def test_v2_csv_contract_rejects_missing_columns(client: httpx.AsyncClient, endpoint: str) -> None:
    response = await client.post(
        endpoint,
        files={"file": ("missing_columns.csv", read_csv_fixture("missing_columns.csv"), "text/csv")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_csv",
            "message": "The CSV upload must include all required measurement columns.",
        }
    }


@pytest.fixture
def safe_surrogate_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a temporary serialised test model without production artifacts."""
    model_directory = tmp_path / "models" / "LR"
    model_directory.mkdir(parents=True)
    artifact_path = model_directory / "test-lr.pkl"
    joblib.dump(ConstantSurrogate(), artifact_path)
    manifest_path = tmp_path / "models" / "production_model_manifest.json"
    artifacts = []
    for model_type in ModelTypeEnum:
        if model_type == ModelTypeEnum.DIRECT_WQI5:
            continue
        relative_path = "models/LR/test-lr.pkl" if model_type == ModelTypeEnum.LR else f"models/{model_type.value}.pkl"
        sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest() if model_type == ModelTypeEnum.LR else "0" * 64
        artifacts.append(
            {
                "model_type": model_type.value,
                "version": "test-1",
                "production_artifact": relative_path,
                "sha256": sha256,
                "training_seed": 0,
                "feature_columns": ["DO", "BOD", "NH3N", "EC", "SS"],
                "evaluation": {"source": "test", "experiment": "contract"},
                "runtime_compatibility": {"scikit_learn": "1.5.2"},
                "metrics": {"mae": 0.0},
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "required_feature_columns": ["DO", "BOD", "NH3N", "EC", "SS"],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    service = WaterQualityService(
        Settings(
            project_root=tmp_path,
            model_dir=tmp_path / "models",
            production_manifest_path=manifest_path,
            default_model=ModelTypeEnum.DIRECT_WQI5,
        )
    )
    monkeypatch.setattr(api, "service", service)


@pytest.mark.anyio
async def test_v2_surrogate_assessment_contract_with_safe_fixture(
    client: httpx.AsyncClient, safe_surrogate_artifact: None
) -> None:
    request = read_json_fixture("single_assessment.json")
    request["model_type"] = "lr"

    response = await client.post("/api/v2/assessment", json=request)

    assert response.status_code == 200
    payload = response.json()
    assert_assessment_contract(payload)
    assert payload["model_type"] == "lr"
    assert payload["score"] == 73.25


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", [
    "/api/v2/assessment", "/predict",
    "/api/v2/assessment/csv/summary", "/api/v2/assessment/csv/rows",
    "/score/total/", "/score/all/",
])
async def test_cold_assessment_keeps_health_responsive_and_loads_once(
    client: httpx.AsyncClient, safe_surrogate_artifact: None,
    monkeypatch: pytest.MonkeyPatch, endpoint: str,
) -> None:
    import asyncio
    import threading

    entered = threading.Event()
    release = threading.Event()
    loads = []
    real_load = joblib.load

    def blocked_load(path):
        loads.append(path)
        entered.set()
        if not release.wait(5):
            raise AssertionError("cold load was not released while health responded")
        return real_load(path)

    monkeypatch.setattr(joblib, "load", blocked_load)

    async def request():
        if endpoint in {"/api/v2/assessment", "/predict"}:
            payload = read_json_fixture("single_assessment.json")
            payload["model_type"] = "lr"
            return await client.post(endpoint, json=payload)
        return await client.post(
            endpoint, data={"model_type": "lr"},
            files={"file": ("input.csv", read_csv_fixture("valid_assessments.csv"), "text/csv")},
        )

    first = asyncio.create_task(request())
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(request())
        health = await asyncio.wait_for(client.get("/api/v2/health"), timeout=1)
        assert health.status_code == 200
        assert not first.done()
        assert len(loads) == 1
    finally:
        release.set()
        responses = await asyncio.gather(first, *([second] if second is not None else []))
    assert len(loads) == 1
    for response in responses:
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_type"] == "lr"
        if endpoint in {"/api/v2/assessment/csv/rows", "/score/all/"}:
            assert payload["scores"] == [73.25, 73.25]
        else:
            assert payload["score"] == 73.25


def test_model_predictions_are_serialized_across_models() -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import pandas as pd

    service = WaterQualityService()
    first_entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()

    class First:
        def predict(self, frame):
            first_entered.set()
            assert release.wait(5)
            return [1.0]

    class Second:
        def predict(self, frame):
            second_entered.set()
            return [2.0]

    service._models = {ModelTypeEnum.LR: First(), ModelTypeEnum.RF: Second()}
    frame = pd.DataFrame([[1, 2, 3, 4, 5]], columns=["DO", "BOD", "NH3N", "EC", "SS"])

    def predict_second():
        second_attempted.set()
        return service._predict(ModelTypeEnum.RF, frame)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service._predict, ModelTypeEnum.LR, frame)
        try:
            assert first_entered.wait(2)
            second = pool.submit(predict_second)
            assert second_attempted.wait(2)
            assert not second_entered.wait(0.1)
        finally:
            release.set()
        assert first.result() == [1.0]
        assert second.result() == [2.0]


@pytest.mark.anyio
@pytest.mark.parametrize("endpoint", ["/api/v2/ready", "/api/v2/models", "/models"])
async def test_concurrent_dependency_checks_hash_once_without_blocking_health(
    client: httpx.AsyncClient, safe_surrogate_artifact: None,
    monkeypatch: pytest.MonkeyPatch, endpoint: str,
) -> None:
    import asyncio
    import threading

    entered = threading.Event()
    release = threading.Event()
    validations = []
    real_pick = api.service._pick_artifact

    def blocked_pick(model_type):
        validations.append(model_type)
        entered.set()
        if not release.wait(5):
            raise AssertionError("artifact hash check blocked health")
        return real_pick(model_type)

    monkeypatch.setattr(api.service, "_pick_artifact", blocked_pick)
    first = asyncio.create_task(client.get(endpoint))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(client.get(endpoint))
        health = await asyncio.wait_for(client.get("/api/v2/health"), timeout=1)
        assert health.status_code == 200
        assert not first.done()
    finally:
        release.set()
        responses = await asyncio.gather(first, *([second] if second is not None else []))
    assert validations == [ModelTypeEnum.LR]
    assert all(response.status_code == 200 for response in responses)
