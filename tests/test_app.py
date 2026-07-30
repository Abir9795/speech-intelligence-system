import io

import pytest

from support import app as app_module


@pytest.fixture()
def client(tmp_path):
    app = app_module.create_app({"TESTING": True, "UPLOAD_FOLDER": tmp_path})
    return app.test_client()


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Speech Recognition" in response.data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "status": "healthy"}


def test_model_info(client):
    response = client.get("/api/model-info")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["command_model"]["classes"] == 10
    assert payload["speaker_model"]["type"] == "closed_set_identification"


def test_upload_requires_file(client):
    response = client.post("/api/predict-file")
    assert response.status_code == 400


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/predict-file",
        data={"audio": (io.BytesIO(b"not audio"), "sample.txt"), "mode": "trained"},
    )
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_upload_uses_inference_result(client, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "run_inference",
        lambda *_args, **_kwargs: {
            "mode": "trained",
            "command_id": "light_on",
            "command_text": "Turn on the light",
            "confidence": 0.9,
            "accepted": True,
        },
    )
    response = client.post(
        "/api/predict-file",
        data={"audio": (io.BytesIO(b"fake wav"), "sample.wav"), "mode": "trained"},
    )
    assert response.status_code == 200
    assert response.get_json()["command_id"] == "light_on"
