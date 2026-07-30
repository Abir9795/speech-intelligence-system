from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .command_model import CommandCNN, CommandDataset, evaluate_model, load_manifest

DEFAULT_EVALUATION_CHECKPOINT = "artifacts/command_model_resplit/best_command_model.pt"


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained speech-command classifier.")
    parser.add_argument("--manifest", default="data/custom_dataset/metadata.csv", help="Path to your metadata CSV.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_EVALUATION_CHECKPOINT,
        help=(
            "Saved checkpoint. Use command_model_resplit for report accuracy; "
            "command_model is the train-all demo model."
        ),
    )
    parser.add_argument("--split", default="test", help="Manifest split to evaluate, or 'all'.")
    parser.add_argument("--speaker-id", help="Optionally evaluate only one speaker ID.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--output-dir", default="reports/command", help="Directory for JSON and CSV reports.")
    parser.add_argument("--show-predictions", action="store_true", help="Print every file-level prediction.")
    return parser


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = base_dir / args.manifest
    checkpoint_path = base_dir / args.checkpoint

    if checkpoint_path.parent.name == "command_model":
        print(
            "WARNING: You are evaluating the train-all demo checkpoint. "
            "Do not report this as held-out test accuracy.\n"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {value: key for key, value in label_to_index.items()}

    rows = load_manifest(manifest_path)
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]
    if args.speaker_id:
        rows = [row for row in rows if row["speaker_id"] == args.speaker_id]
    if not rows:
        raise ValueError(f"No rows found for split '{args.split}'.")

    dataset = CommandDataset(rows, label_to_index, base_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommandCNN(num_classes=len(label_to_index)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = nn.CrossEntropyLoss()

    metrics = evaluate_model(model, loader, loss_fn, device)
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    predictions: list[dict[str, str]] = []
    model.eval()
    with torch.no_grad():
        for row, (features, _) in zip(rows, loader.dataset, strict=True):
            logits = model(features.unsqueeze(0).to(device))
            predicted_index = int(logits.argmax(dim=1).item())
            predicted_label = index_to_label[predicted_index]
            true_labels.append(row["command_id"])
            predicted_labels.append(predicted_label)
            predictions.append(
                {"audio_path": row["audio_path"], "true_label": row["command_id"], "predicted_label": predicted_label}
            )
            if args.show_predictions:
                print(f"FILE: {row['audio_path']}")
                print(f"TRUE: {row['command_id']}")
                print(f"PRED: {predicted_label}")
                print("-" * 40)

    from sklearn.metrics import classification_report, confusion_matrix

    labels = sorted(label_to_index, key=label_to_index.get)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_path.relative_to(base_dir)),
        "manifest": str(manifest_path.relative_to(base_dir)),
        "split": args.split,
        "speaker_id": args.speaker_id,
        "samples": len(rows),
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "classification_report": report,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(output_dir / "predictions.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_path", "true_label", "predicted_label"])
        writer.writeheader()
        writer.writerows(predictions)
    with open(output_dir / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *labels])
        for label, values in zip(labels, matrix, strict=True):
            writer.writerow([label, *values.tolist()])

    print(f"Accuracy on split '{args.split}': {metrics.accuracy:.4f}")
    print(f"Macro F1 on split '{args.split}': {report['macro avg']['f1-score']:.4f}")
    print(f"Loss on split '{args.split}': {metrics.loss:.4f}")
    print(f"Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
