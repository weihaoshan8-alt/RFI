"""Step 1: V13 open-set routing and export of accepted IQ blocks.

This is the software-team entry point for making a new stage-2 input dataset.
Only IQ is required in the source NPZ.  ``labels`` and ``class_names`` are
optional and are copied only when they are available for offline evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from recognition_core import V13RecognitionService
from recognition_core.core import _as_iq_channels, _fit_block_length


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 open-set extraction of accepted IQ blocks")
    # ===== Software integration: these three defaults are the usual edit points. =====
    parser.add_argument("--input", type=Path, default=ROOT / "test_data" / "recognition_test_v13_formal_4000.npz", help="source IQ NPZ; requires an iq field")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "software_output", help="directory for the newly generated datasets and records")
    parser.add_argument("--accepted-name", default="stage1_accepted_known_iq.npz", help="name of the newly generated accepted-IQ dataset")
    # ================================================================================
    parser.add_argument("--rejected-name", default="stage1_rejected_unknown_iq.npz", help="name of the exported rejected-IQ dataset")
    parser.add_argument("--records-name", default="stage1_open_set_records.csv", help="name of the per-block result CSV")
    parser.add_argument("--summary-name", default="stage1_open_set_summary.json", help="name of the stage-1 result JSON")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "v13_open_set_model.npz")
    parser.add_argument("--cluster-unknown", action="store_true", help="also calculate unknown-cluster IDs; useful for batch test display")
    return parser.parse_args()


def write_npz(path: Path, blocks: np.ndarray, indices: np.ndarray, records: list[dict], source: dict) -> None:
    payload: dict[str, np.ndarray] = {
        "iq": blocks.astype(np.float32),
        "source_block_indices": indices.astype(np.int64),
        "stage1_knownness_score": np.asarray([item["knownness_score"] for item in records], dtype=np.float32),
        "stage1_closest_known_class": np.asarray([item["closest_known_class"] for item in records]),
        "metadata": np.asarray(json.dumps({
            "purpose": "V13 stage-1 accepted/rejected IQ extraction",
            "labels_usage": "optional offline evaluation only; never used by V13 or V20 inference",
        }, ensure_ascii=False), dtype=np.str_),
    }
    # Preserve ground truth only for a labelled formal/test NPZ.  It is never
    # accessed by inference, but makes later accuracy and confusion-matrix
    # calculation possible.
    labels = source.get("labels")
    if labels is not None and len(labels) == len(source["iq"]):
        payload["labels"] = np.asarray(labels, dtype=np.int64)[indices]
    if source.get("class_names") is not None:
        payload["class_names"] = np.asarray(source["class_names"])
    np.savez_compressed(path, **payload)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input, allow_pickle=False) as source_file:
        source = {name: np.asarray(source_file[name]) for name in source_file.files}
    if "iq" not in source:
        raise KeyError("source NPZ must contain an iq field")
    blocks = _fit_block_length(_as_iq_channels(source["iq"]), 16384)
    result = V13RecognitionService(args.model).predict_array(blocks, cluster_unknown=args.cluster_unknown)
    records = result["records"]
    accepted = np.asarray([row["is_known"] for row in records], dtype=bool)
    accepted_indices = np.flatnonzero(accepted)
    rejected_indices = np.flatnonzero(~accepted)
    accepted_path = args.output_dir / args.accepted_name
    rejected_path = args.output_dir / args.rejected_name
    write_npz(accepted_path, blocks[accepted], accepted_indices, [records[i] for i in accepted_indices], source)
    write_npz(rejected_path, blocks[~accepted], rejected_indices, [records[i] for i in rejected_indices], source)
    record_path = args.output_dir / args.records_name
    fields = ("block_index", "is_known", "label", "closest_known_class", "knownness_score", "unknown_cluster_id")
    with record_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in records])
    summary = {
        "source": str(args.input.resolve()), "model": str(args.model.resolve()),
        "input_block_count": int(len(blocks)), "accepted_block_count": int(np.sum(accepted)),
        "rejected_block_count": int(np.sum(~accepted)), "accepted_dataset": str(accepted_path.resolve()),
        "rejected_dataset": str(rejected_path.resolve()), "records": str(record_path.resolve()),
        "cluster_unknown": bool(args.cluster_unknown),
    }
    summary_path = args.output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("V13 input blocks:", len(blocks))
    print("V13 accepted known blocks:", int(np.sum(accepted)))
    print("V13 rejected unknown blocks:", int(np.sum(~accepted)))
    print("Accepted IQ dataset:", accepted_path)
    print("Rejected IQ dataset:", rejected_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
