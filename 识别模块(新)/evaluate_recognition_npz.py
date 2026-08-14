"""运行 V13 正式 NPZ 测试并生成软件可读取的结果和两张展示图。"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "reference_results" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from recognition_core import V13RecognitionService
from recognition_core.v13_analog_cluster import analog_modulation_features


DEFAULT_MODEL = ROOT / "model" / "v13_open_set_model.npz"
DEFAULT_DATASET = ROOT / "test_data" / "recognition_test_v13_formal_4000.npz"
DEFAULT_OUTPUT = ROOT / "reference_results"


def _json_ready(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"records", "roc_curve"}
    }


def _save_records(path: Path, records) -> None:
    fields = (
        "block_index",
        "is_known",
        "label",
        "closest_known_class",
        "knownness_score",
        "unknown_cluster_id",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def _plot_open_confusion(result: Dict[str, Any], path: Path) -> None:
    matrix = np.asarray(
        result["open_set_confusion_matrix"], dtype=np.int64
    )
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(8.6, 7.0), dpi=160)
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks([0, 1], labels=["Predicted known", "Predicted unknown"])
    axis.set_yticks([0, 1], labels=["Actual known", "Actual unknown"])
    axis.set_xlabel("Model decision")
    axis.set_ylabel("Ground truth")
    axis.set_title("Open-set known / unknown confusion matrix")
    for row in range(2):
        for column in range(2):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                "{:.2f}%\n({:,} samples)".format(
                    value * 100.0, matrix[row, column]
                ),
                ha="center",
                va="center",
                color="white" if value > 0.5 else "#10243e",
                fontsize=12,
                fontweight="bold",
            )
    fig.colorbar(image, ax=axis, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(result: Dict[str, Any], path: Path) -> None:
    roc = result["roc_curve"]
    fpr = np.asarray(roc["false_positive_rate"], dtype=np.float64)
    tpr = np.asarray(roc["true_positive_rate"], dtype=np.float64)
    auroc = float(result["display_metrics"]["auroc"])
    fig, axis = plt.subplots(figsize=(8.6, 7.0), dpi=160)
    axis.plot(
        fpr,
        tpr,
        color="#0878c9",
        linewidth=2.5,
        label="V13 ROC (AUROC = {:.5f})".format(auroc),
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#808080",
        linewidth=1.4,
        label="Random baseline",
    )
    axis.fill_between(fpr, tpr, alpha=0.15, color="#0878c9")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("False acceptance rate of unknown signals")
    axis.set_ylabel("Acceptance rate of known signals")
    axis.set_title("Open-set recognition ROC curve")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_unknown_cluster_feature_space(
    dataset_path: Path,
    result: Dict[str, Any],
    path: Path,
) -> None:
    """Draw rejected unknown samples in a 2-D PCA feature space.

    The V13 GMM clustering has already been completed inside ``evaluate_npz``.
    This function only visualizes its assigned cluster IDs.  Ground-truth
    AM/FM labels are used exclusively to colour an *offline test plot* and
    are never supplied to the V13 decision or GMM fitting process.
    """
    with np.load(dataset_path, allow_pickle=False) as source:
        iq = np.asarray(source["iq"], dtype=np.float32)
        labels = np.asarray(source["labels"], dtype=np.int64)
        class_names = np.asarray(source["class_names"])
    records = result["records"]
    rejected_indices = np.asarray(
        [index for index, row in enumerate(records) if not row["is_known"]],
        dtype=np.int64,
    )
    if len(rejected_indices) < 2:
        return
    # For this offline *unknown-cluster* visualization, retain true target
    # unknowns only.  A few true-known blocks can be rejected by an open-set
    # detector; showing them here would visually contaminate the AM/FM clusters.
    # This is display filtering only, performed after all V13/GMM decisions.
    all_actual_names = np.asarray([str(class_names[item]) for item in labels])
    target_unknown = ~np.isin(
        all_actual_names[rejected_indices],
        np.asarray(result["known_classes"]),
    )
    rejected_indices = rejected_indices[target_unknown]
    if len(rejected_indices) < 2:
        return
    features = analog_modulation_features(iq[rejected_indices])
    coordinates = PCA(n_components=2, random_state=20260810).fit_transform(
        StandardScaler().fit_transform(features)
    )
    actual_names = all_actual_names[rejected_indices]
    cluster_ids = np.asarray(
        [records[index]["unknown_cluster_id"] for index in rejected_indices],
        dtype=np.int64,
    )
    markers = {0: "o", 1: "^"}
    colours = {"AM": "#d1495b", "FM": "#157da8"}
    figure, axis = plt.subplots(figsize=(10.2, 7.4), dpi=160)
    for actual_name in sorted(np.unique(actual_names)):
        for cluster_id in sorted(np.unique(cluster_ids)):
            selected = (actual_names == actual_name) & (cluster_ids == cluster_id)
            if not np.any(selected):
                continue
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=20,
                marker=markers.get(int(cluster_id), "s"),
                color=colours.get(str(actual_name), "#555555"),
                alpha=0.72,
                label="{} / V13 cluster {}".format(actual_name, int(cluster_id)),
            )
    axis.set_title("Unknown interference feature space after V13 rejection")
    axis.set_xlabel("Feature dimension 1 (PCA projection)")
    axis.set_ylabel("Feature dimension 2 (PCA projection)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def evaluate(
    dataset_path: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    """供 PyQt/PySide 调用；返回指标、记录和两张结果图路径。"""
    dataset_path = Path(dataset_path).resolve()
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    service = V13RecognitionService(model_path)
    result = service.evaluate_npz(dataset_path)
    confusion_path = output_dir / "open_set_confusion_matrix.png"
    roc_path = output_dir / "open_set_roc_curve.png"
    cluster_path = output_dir / "unknown_cluster_feature_space.png"
    summary_path = output_dir / "recognition_summary.json"
    records_path = output_dir / "recognition_records.csv"
    _plot_open_confusion(result, confusion_path)
    _plot_roc(result, roc_path)
    _plot_unknown_cluster_feature_space(dataset_path, result, cluster_path)
    summary = _json_ready(result)
    summary["confusion_matrix_plot_path"] = str(
        confusion_path.relative_to(ROOT)
    )
    summary["roc_curve_plot_path"] = str(roc_path.relative_to(ROOT))
    summary["unknown_cluster_feature_space_plot_path"] = str(
        cluster_path.relative_to(ROOT)
    )
    summary["records_csv_path"] = str(records_path.relative_to(ROOT))
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_records(records_path, result["records"])
    result["summary_json_path"] = str(summary_path)
    result["records_csv_path"] = str(records_path)
    result["confusion_matrix_plot_path"] = str(confusion_path)
    result["roc_curve_plot_path"] = str(roc_path)
    result["unknown_cluster_feature_space_plot_path"] = str(cluster_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V13 严格开放集识别 NPZ 测试"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.model, args.output_dir)
    display = result["display_metrics"]
    print("=" * 76)
    print("V13 strict open-set recognition")
    print(
        "Known signal recognition rate : {:.4f}%".format(
            100.0 * display["known_signal_recognition_rate"]
        )
    )
    print(
        "Unknown rejection rate         : {:.4f}%".format(
            100.0 * display["unknown_rejection_rate"]
        )
    )
    print("AUROC                         : {:.6f}".format(display["auroc"]))
    print(
        "Unknown clustering NMI         : {:.6f}".format(
            display["unknown_clustering_nmi"]
        )
    )
    print(
        "Unknown clustering accuracy    : {:.4f}%".format(
            100.0 * display["unknown_clustering_accuracy"]
        )
    )
    print("Summary:", result["summary_json_path"])
    print("Plot   :", result["confusion_matrix_plot_path"])
    print("Plot   :", result["roc_curve_plot_path"])
    print("Plot   :", result["unknown_cluster_feature_space_plot_path"])
    print("=" * 76)


if __name__ == "__main__":
    main()
