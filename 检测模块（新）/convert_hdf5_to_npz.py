"""把项目 HDF5 数据集转换为软件逐类读取的 NPZ 数据包。

如果已经拿到完整的 detection_dataset_formal_npz 文件夹，
软件运行时不需要执行本脚本。本脚本仅用于重新制作 NPZ 数据集。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from core import SIGNAL_TYPES, calculate_block_energies


def dataset_energies(dataset: h5py.Dataset, batch_size: int = 64) -> np.ndarray:
    """分批调用 core.calculate_block_energies 计算 HDF5 噪声能量。"""
    energies = np.empty(dataset.shape[0], dtype=np.float64)
    for start in range(0, dataset.shape[0], batch_size):
        stop = min(dataset.shape[0], start + batch_size)
        energies[start:stop] = calculate_block_energies(dataset[start:stop])
    return energies


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def convert(input_path: Path, output_dir: Path, batch_size: int = 64) -> Dict:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict = {
        "format": "rfi-detection-npz-v1",
        "source_hdf5": str(input_path),
        "signal_types": [],
        "files": {},
        "notes": [
            "noise.npz stores H0 calibration and validation energies.",
            "Each signal NPZ stores raw H1 IQ as [block,2,sample] float32.",
        ],
    }

    with h5py.File(str(input_path), "r") as source:
        attrs = {str(key): _json_value(value) for key, value in source.attrs.items()}
        manifest["attributes"] = attrs

        calibration = dataset_energies(source["noise/calibration/iq"], batch_size)
        validation = dataset_energies(source["noise/validation/iq"], batch_size)
        noise_path = output_dir / "noise.npz"
        np.savez(
            str(noise_path),
            calibration_energy=calibration,
            validation_energy=validation,
            pfa_target=np.asarray(
                float(attrs.get("pfa_target", 0.05)), dtype=np.float64
            ),
            sample_count=np.asarray(
                int(attrs.get("sample_count", 0)), dtype=np.int64
            ),
            sample_rate_hz=np.asarray(
                float(attrs.get("sample_rate_hz", 0.0)), dtype=np.float64
            ),
            snr_db=np.asarray(float(attrs.get("snr_db", np.nan)), dtype=np.float64),
        )
        manifest["files"]["noise"] = {
            "path": noise_path.name,
            "calibration_blocks": int(calibration.size),
            "validation_blocks": int(validation.size),
        }

        rows: List[Dict] = []
        for signal_type in SIGNAL_TYPES:
            group_path = "signals/{}".format(signal_type)
            if group_path not in source:
                continue
            group = source[group_path]
            repeat_names = sorted(group.keys())
            iq_arrays: List[np.ndarray] = []
            repeat_indexes: List[np.ndarray] = []

            for repeat_index, repeat_name in enumerate(repeat_names, start=1):
                iq = group["{}/iq".format(repeat_name)][:].astype(
                    np.float32, copy=False
                )
                iq_arrays.append(iq)
                repeat_indexes.append(
                    np.full(iq.shape[0], repeat_index, dtype=np.int16)
                )

            all_iq = np.concatenate(iq_arrays, axis=0)
            all_repeat_indexes = np.concatenate(repeat_indexes, axis=0)
            path = output_dir / "{}.npz".format(signal_type)
            np.savez(
                str(path),
                iq=all_iq,
                repeat_index=all_repeat_indexes,
                signal_type=np.asarray(signal_type),
                sample_count=np.asarray(all_iq.shape[-1], dtype=np.int64),
                sample_rate_hz=np.asarray(
                    float(attrs.get("sample_rate_hz", 0.0)), dtype=np.float64
                ),
                snr_db=np.asarray(
                    float(group.attrs.get("snr_db", attrs.get("snr_db", np.nan))),
                    dtype=np.float64,
                ),
            )

            manifest["signal_types"].append(signal_type)
            manifest["files"][signal_type] = {
                "path": path.name,
                "blocks": int(all_iq.shape[0]),
                "repeat_count": len(repeat_names),
            }
            for repeat_index, repeat_name in enumerate(repeat_names, start=1):
                rows.append(
                    {
                        "signal_type": signal_type,
                        "repeat": repeat_index,
                        "blocks": int(group["{}/iq".format(repeat_name)].shape[0]),
                        "npz_file": path.name,
                    }
                )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["signal_type", "repeat", "blocks", "npz_file"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="HDF5 转逐类 NPZ")
    parser.add_argument("--input", type=Path, required=True, help="输入 HDF5 文件")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    manifest = convert(args.input, args.output_dir, args.batch_size)
    print("NPZ dataset created successfully")
    print("Output:", args.output_dir.resolve())
    print("Signals:", ", ".join(manifest["signal_types"]))


if __name__ == "__main__":
    main()
