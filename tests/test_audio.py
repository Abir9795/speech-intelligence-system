import wave

import numpy as np

from support.command_model import SAMPLE_RATE, fix_length, prepare_audio, read_wav_mono
from support.transcriber import allowed_audio_file


def test_fix_length_pads_and_truncates():
    short = np.ones(100, dtype=np.float32)
    long = np.ones(SAMPLE_RATE * 4, dtype=np.float32)

    assert fix_length(short).shape == (SAMPLE_RATE * 3,)
    assert fix_length(long).shape == (SAMPLE_RATE * 3,)


def test_prepare_audio_returns_finite_fixed_length_audio():
    audio = np.concatenate([np.zeros(2000), np.ones(4000) * 0.02]).astype(np.float32)
    prepared = prepare_audio(audio)

    assert prepared.shape == (SAMPLE_RATE * 3,)
    assert np.isfinite(prepared).all()


def test_read_wav_mono(tmp_path):
    wav_path = tmp_path / "sample.wav"
    samples = (np.sin(np.linspace(0, 20, 8000)) * 10_000).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())

    audio = read_wav_mono(wav_path)

    assert len(audio) == 16000
    assert audio.dtype == np.float32


def test_allowed_audio_extensions_are_explicit():
    assert allowed_audio_file("voice.wav")
    assert allowed_audio_file("voice.MP3")
    assert not allowed_audio_file("voice.txt")
    assert not allowed_audio_file("voice.wav.exe")
