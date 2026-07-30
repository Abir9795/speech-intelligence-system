from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from flask import Blueprint, Flask, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from .command_model import DEFAULT_CHECKPOINT, predict_command
from .speaker_model import DEFAULT_SPEAKER_MODEL, predict_speaker
from .transcriber import DEFAULT_MODEL, allowed_audio_file, transcribe_file

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEB_PORT = 5050
DEFAULT_WHISPER_LANGUAGE = "en"
ALLOWED_WHISPER_MODELS = {"tiny", "base", "small", "medium"}
logger = logging.getLogger(__name__)
web = Blueprint("web", __name__)


def create_app(config: dict | None = None) -> Flask:
    flask_app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    flask_app.config.from_mapping(
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024,
        UPLOAD_FOLDER=BASE_DIR / "uploads",
        CHECKPOINT_PATH=Path(os.getenv("COMMAND_CHECKPOINT", BASE_DIR / DEFAULT_CHECKPOINT)),
        SPEAKER_MODEL_PATH=Path(os.getenv("SPEAKER_MODEL", BASE_DIR / DEFAULT_SPEAKER_MODEL)),
        COMMAND_CONFIDENCE_THRESHOLD=float(os.getenv("COMMAND_CONFIDENCE_THRESHOLD", "0.55")),
    )
    if config:
        flask_app.config.update(config)
    Path(flask_app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    flask_app.register_blueprint(web)
    return flask_app


@web.get("/")
def index():
    return render_template(
        "index.html",
        checkpoint_path=str(Path(current_app.config["CHECKPOINT_PATH"]).relative_to(BASE_DIR)),
        default_model=DEFAULT_MODEL,
    )


@web.get("/health")
def health():
    return jsonify({"ok": True, "status": "healthy"})


@web.get("/api/model-info")
def model_info():
    checkpoint_path = Path(current_app.config["CHECKPOINT_PATH"])
    speaker_model_path = Path(current_app.config["SPEAKER_MODEL_PATH"])
    return jsonify(
        {
            "ok": True,
            "command_model": {"available": checkpoint_path.exists(), "classes": 10},
            "speaker_model": {"available": speaker_model_path.exists(), "type": "closed_set_identification"},
            "whisper": {"default_model": DEFAULT_MODEL, "allowed_models": sorted(ALLOWED_WHISPER_MODELS)},
        }
    )


def error_response(message: str, status_code: int = 400):
    return jsonify({"ok": False, "error": message}), status_code


def run_inference(audio_path: Path, mode: str, whisper_model: str = DEFAULT_MODEL):
    speaker_prediction = predict_speaker(audio_path, model_path=current_app.config["SPEAKER_MODEL_PATH"])

    if mode == "trained":
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("The trained command model only supports WAV input.")

        prediction = predict_command(audio_path, checkpoint_path=current_app.config["CHECKPOINT_PATH"])
        payload = {
            "mode": "trained",
            "model": "trained_command_model",
            "command_id": prediction["command_id"],
            "command_text": prediction["command_text"],
            "confidence": prediction["confidence"],
            "accepted": prediction["confidence"] >= current_app.config["COMMAND_CONFIDENCE_THRESHOLD"],
            "confidence_threshold": current_app.config["COMMAND_CONFIDENCE_THRESHOLD"],
        }
        payload.update(speaker_prediction)
        return payload

    if mode == "whisper":
        if whisper_model not in ALLOWED_WHISPER_MODELS:
            raise ValueError(f"Unsupported Whisper model: {whisper_model}")
        transcript = transcribe_file(audio_path, model_name=whisper_model, language=DEFAULT_WHISPER_LANGUAGE)
        payload = {
            "mode": "whisper",
            "model": whisper_model,
            "transcript": transcript,
        }
        payload.update(speaker_prediction)
        return payload

    raise ValueError("Unsupported mode. Choose 'trained' or 'whisper'.")


@web.post("/api/predict-file")
def predict_uploaded_file():
    audio_file = request.files.get("audio")
    mode = request.form.get("mode", "trained").strip().lower()
    whisper_model = request.form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if audio_file is None or audio_file.filename == "":
        return error_response("Please choose an audio file first.")

    if not allowed_audio_file(audio_file.filename):
        return error_response("Unsupported file type. Upload WAV, MP3, M4A, MP4, or WebM audio.")

    safe_name = secure_filename(audio_file.filename)
    suffix = Path(safe_name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=current_app.config["UPLOAD_FOLDER"]) as temp_file:
        target_path = Path(temp_file.name)

    try:
        audio_file.save(target_path)
        payload = run_inference(target_path, mode=mode, whisper_model=whisper_model)
        payload.update({"ok": True, "filename": audio_file.filename})
        return jsonify(payload)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception:
        logger.exception("Audio file inference failed")
        return error_response("The audio could not be processed.", 500)
    finally:
        target_path.unlink(missing_ok=True)


@web.post("/api/predict-recording")
def predict_recording():
    audio_file = request.files.get("audio")
    mode = request.form.get("mode", "trained").strip().lower()
    whisper_model = request.form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if audio_file is None:
        return error_response("No recording was received.")

    suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=current_app.config["UPLOAD_FOLDER"],
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        audio_file.save(temp_path)
        payload = run_inference(temp_path, mode=mode, whisper_model=whisper_model)
        payload.update({"ok": True, "filename": audio_file.filename or "recording.wav"})
        return jsonify(payload)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception:
        logger.exception("Recorded audio inference failed")
        return error_response("The recording could not be processed.", 500)
    finally:
        temp_path.unlink(missing_ok=True)


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=DEFAULT_WEB_PORT, debug=False, use_reloader=False)
