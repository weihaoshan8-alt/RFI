"""Formal V13-open-set + V20-closed-set evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "reference_results" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from recognition_core import TwoStageV20RecognitionService


ROOT = Path(__file__).resolve().parent
KNOWN = ("2ASK", "2FSK", "BPSK", "16QAM")


def save_matrix(matrix: np.ndarray, path: Path) -> None:
    ratio = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(7.2, 6.1), dpi=160)
    image = axis.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(4), KNOWN); axis.set_yticks(range(4), KNOWN)
    axis.set_xlabel("V20 closed-set prediction"); axis.set_ylabel("Actual known class")
    axis.set_title("V13 accepted groups → V20 four-class recognition")
    for row in range(4):
        for col in range(4):
            axis.text(col, row, "{:.1f}%\n({})".format(100 * ratio[row, col], matrix[row, col]), ha="center", va="center", color="white" if ratio[row, col] > .55 else "black")
    figure.colorbar(image, ax=axis, label="Row-normalized ratio"); figure.tight_layout(); figure.savefig(path, bbox_inches="tight"); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "test_data" / "recognition_test_v13_formal_4000.npz")
    parser.add_argument("--open-model", type=Path, default=ROOT / "model" / "v13_open_set_model.npz")
    parser.add_argument("--closed-model", type=Path, default=ROOT / "model" / "v20_closed_set_long_window_model.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reference_results" / "v20_two_stage")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as handle:
        iq = np.asarray(handle["iq"], dtype=np.float32); labels = np.asarray(handle["labels"], dtype=np.int64); names = np.asarray(handle["class_names"])
    actual = np.asarray([str(names[item]) for item in labels]); known = np.isin(actual, KNOWN)
    service = TwoStageV20RecognitionService(args.open_model, args.closed_model)
    output = service.predict_array(iq, cluster_unknown=False)
    stage1 = output["stage1"]; accepted = np.asarray([row["is_known"] for row in stage1["records"]], dtype=bool); accepted_indices = np.flatnonzero(accepted)
    stage2 = output["stage2"]; size = int(stage2["group_size"]); used = int(stage2["used_blocks"])
    groups = accepted_indices[:used].reshape(-1, size); group_actual = actual[groups]
    homogeneous = np.asarray([np.all(row == row[0]) and row[0] in KNOWN for row in group_actual], dtype=bool)
    true = np.asarray([KNOWN.index(row[0]) for row in group_actual[homogeneous]], dtype=np.int64)
    predicted_names = np.asarray([row["closed_set_label"] for row in stage2["records"]]); predicted = np.asarray([KNOWN.index(item) for item in predicted_names[homogeneous]], dtype=np.int64)
    matrix = confusion_matrix(true, predicted, labels=np.arange(4)); accuracy = float(np.mean(true == predicted))
    matrix_path = args.output_dir / "v20_known_four_class_confusion_matrix.png"; save_matrix(matrix, matrix_path)
    rows = []
    for group_id, indexes in enumerate(groups):
        values = group_actual[group_id]
        rows.append({"group_id": group_id, "source_block_start": int(indexes[0]), "source_block_end": int(indexes[-1]), "actual_group_label_for_offline_metric": str(values[0]) if np.all(values == values[0]) else "MIXED", "is_homogeneous_known_group": bool(homogeneous[group_id]), "v20_prediction": str(predicted_names[group_id])})
    record_path = args.output_dir / "v20_group_records.csv"
    with record_path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    report = {"protocol": "strict_open_set_v13_then_known_only_v20_closed_set", "unknown_training_or_selection_samples_read": 0, "known_acceptance_rate": float(np.mean(accepted[known])), "unknown_rejection_rate": float(np.mean(~accepted[~known])), "stage2_group_size_blocks": size, "stage2_group_iq_points": size * 16384, "stage2_homogeneous_known_groups": int(np.sum(homogeneous)), "stage2_four_class_accuracy": accuracy, "confusion_matrix": matrix.tolist(), "matrix": str(matrix_path.resolve()), "records": str(record_path.resolve())}
    report_path = args.output_dir / "v20_two_stage_summary.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("V13 known acceptance: {:.4f}%".format(100 * report["known_acceptance_rate"])); print("V13 unknown rejection: {:.4f}%".format(100 * report["unknown_rejection_rate"])); print("V20 four-class accuracy: {:.4f}%".format(100 * accuracy)); print("Saved:", matrix_path); print("Saved:", report_path)


if __name__ == "__main__":
    main()
