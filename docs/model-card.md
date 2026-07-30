# Model Card

## Summary

This repository contains two custom models:

1. A CNN that classifies ten English household-style voice commands.
2. A linear SVM that identifies one of ten enrolled speakers.

Whisper Base is included as a general speech-to-text baseline.

## Dataset

The current manifest contains 300 English WAV recordings:

- 10 speakers
- 10 commands
- 3 takes per speaker-command pair
- 30 examples per command

The dataset is balanced by speaker and command. Its small size and controlled collection setup limit real-world generalization.

## Evaluation

### Command model

The shipped evaluation checkpoint achieves:

- 65% accuracy
- 100 held-out third-take recordings
- all ten speakers are represented in training and test

This is a known-speaker, new-take evaluation. It must not be described as unseen-speaker performance.

The training CLI also supports speaker-disjoint evaluation:

```bash
python -m support.train_command_model \
  --split-by-speaker \
  --val-speakers speaker_09 \
  --test-speakers speaker_10 \
  --output-dir artifacts/command_model_speaker_disjoint
```

An included speaker-disjoint checkpoint trained on speakers 1–8, validated on speaker 9, and tested on speaker 10 achieves 33.3% accuracy on the 30 recordings from speaker 10. This sharp drop from the known-speaker result is evidence that more speaker diversity would be needed for robust deployment.

### Speaker model

The shipped model achieves 98% accuracy on 100 held-out third-take recordings. The identities in the test set are enrolled identities seen during training, so this is closed-set speaker identification.

## Intended use

- Educational experimentation
- Comparing command classification and transcription
- Demonstrating a small end-to-end audio ML application

## Unsupported use

- Authentication or identity verification
- Safety-critical commands
- Surveillance
- Production decisions about people
- Languages or command phrases outside the documented dataset

## Limitations

- Very small dataset
- Limited recording conditions and microphones
- No unknown-speaker rejection in the shipped speaker model
- Softmax/SVM probabilities are not calibrated guarantees
- The command model always chooses one of ten labels
- No systematic noisy-room evaluation yet

## Reproducibility

Use Python 3.10–3.12. Project dependency ranges are declared in `pyproject.toml`. The persisted SVM should be retrained when the scikit-learn version changes because pickle/joblib model compatibility is not guaranteed across versions.

## Privacy

Voice recordings can be biometric and personally identifying. See `docs/privacy.md` before publishing or redistributing the dataset.
