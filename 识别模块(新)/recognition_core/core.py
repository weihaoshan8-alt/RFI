"""V13 严格开放集 IQ 识别核心接口。

本文件不依赖 PyTorch，也不需要 GPU。软件初始化一次服务后，可以反复调用
``predict_array``、``predict_file`` 或 ``evaluate_npz``。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT",
    str(max(1, os.cpu_count() or 1)),
)

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .metrics import clustering_metrics
from .v6_nonlinear import load_v6_artifact, predict_v6_artifact
from .v7_cumulant import v7_features
from .v13_analog_cluster import analog_modulation_features


def _as_iq_channels(value: np.ndarray) -> np.ndarray:
    """把常见 IQ 排列统一为 ``[B,2,N] float32``。"""
    value = np.asarray(value)
    if value.size == 0:
        raise ValueError("IQ 数据不能为空")
    if not np.all(np.isfinite(value)):
        raise ValueError("IQ 数据不能包含 NaN 或无穷大")
    if np.iscomplexobj(value):
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2:
            raise ValueError("复数 IQ 必须为 [N] 或 [B,N]")
        return np.stack((value.real, value.imag), axis=1).astype(np.float32)
    value = value.astype(np.float32, copy=False)
    if value.ndim == 2 and value.shape[0] == 2:
        return value[None, :, :]
    if value.ndim == 3 and value.shape[1] == 2:
        return value
    if value.ndim == 2 and value.shape[1] == 2:
        return value.T[None, :, :]
    if value.ndim == 3 and value.shape[2] == 2:
        return np.transpose(value, (0, 2, 1))
    raise ValueError(
        "实数 IQ 必须为 [2,N]、[N,2]、[B,2,N] 或 [B,N,2]"
    )


def _fit_block_length(iq: np.ndarray, sample_count: int) -> np.ndarray:
    """补零或分段，使每个模型输入块恰好包含 sample_count 个采样点。"""
    if iq.shape[-1] == sample_count:
        return iq
    if iq.shape[-1] < sample_count:
        padding = sample_count - iq.shape[-1]
        return np.pad(iq, ((0, 0), (0, 0), (0, padding)))
    blocks: List[np.ndarray] = []
    for batch_item in iq:
        for start in range(0, batch_item.shape[-1], sample_count):
            block = batch_item[:, start : start + sample_count]
            if block.shape[-1] < sample_count:
                block = np.pad(
                    block,
                    ((0, 0), (0, sample_count - block.shape[-1])),
                )
            blocks.append(block)
    return np.stack(blocks).astype(np.float32)


def load_iq_file(
    path: Path,
    hdf5_dataset: Optional[str] = None,
    dat_dtype: str = "complex64",
) -> np.ndarray:
    """读取 NPY、NPZ、HDF5 或 DAT IQ 文件。"""
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "iq" if "iq" in archive.files else archive.files[0]
            return np.asarray(archive[key])
    if suffix in (".h5", ".hdf5"):
        if not hdf5_dataset:
            raise ValueError("读取 HDF5 时必须指定 hdf5_dataset")
        with h5py.File(path, "r") as handle:
            return np.asarray(handle[hdf5_dataset])
    if suffix == ".dat":
        return np.fromfile(path, dtype=np.dtype(dat_dtype))
    raise ValueError("不支持的 IQ 文件格式：{}".format(suffix))


def _oscr_scalar(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_correct: np.ndarray,
) -> float:
    thresholds = np.r_[
        np.inf,
        np.sort(np.unique(np.r_[known_scores, unknown_scores]))[::-1],
        -np.inf,
    ]
    false_positive_rate = np.asarray(
        [np.mean(unknown_scores >= value) for value in thresholds],
        dtype=np.float64,
    )
    correct_classification_rate = np.asarray(
        [
            np.mean(known_correct & (known_scores >= value))
            for value in thresholds
        ],
        dtype=np.float64,
    )
    order = np.argsort(false_positive_rate, kind="stable")
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(
        integrate(
            correct_classification_rate[order],
            false_positive_rate[order],
        )
    )


class V13RecognitionService:
    """已知数字调制识别、目标未知拒识与未知 AM/FM 聚类服务。"""

    version = "V13"

    def __init__(self, model_path: Path, sample_count: int = 16384) -> None:
        self.model_path = Path(model_path).resolve()
        self.sample_count = int(sample_count)
        self.artifact = load_v6_artifact(self.model_path)
        self.metadata = self.artifact["metadata"]
        if self.metadata.get("protocol") != "strict_open_set_known_only":
            raise RuntimeError("模型不是 strict_open_set_known_only 模型")
        self.class_names = list(self.metadata["known_classes"])

    def _cluster_rejected(
        self,
        blocks: np.ndarray,
        rejected: np.ndarray,
    ) -> Tuple[np.ndarray, str]:
        cluster_ids = np.full(len(blocks), -1, dtype=np.int64)
        rejected_count = int(np.sum(rejected))
        if rejected_count < 2:
            return cluster_ids, "至少需要两个被拒识 IQ 块才能聚类"
        features = analog_modulation_features(blocks[rejected])
        scaled = StandardScaler().fit_transform(features)
        cluster_ids[rejected] = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=1e-4,
            n_init=30,
            random_state=20260806,
        ).fit_predict(scaled)
        warning = ""
        if rejected_count < 500:
            warning = (
                "被拒识样本少于500条，两个未知簇的稳定性可能不足；"
                "该数值是当前数据上的工程建议。"
            )
        return cluster_ids, warning

    def predict_array(
        self,
        iq: np.ndarray,
        cluster_unknown: bool = True,
    ) -> Dict[str, Any]:
        """对一个或一批 IQ 块执行识别、拒识和可选未知聚类。"""
        blocks = _fit_block_length(
            _as_iq_channels(iq), self.sample_count
        )
        features = v7_features(blocks)
        prediction, score, distances, _ = predict_v6_artifact(
            features, self.artifact
        )
        rejected = score < 0.0
        cluster_ids = np.full(len(blocks), -1, dtype=np.int64)
        cluster_warning = ""
        if cluster_unknown:
            cluster_ids, cluster_warning = self._cluster_rejected(
                blocks, rejected
            )
        records: List[Dict[str, Any]] = []
        for index in range(len(blocks)):
            class_index = int(prediction[index])
            is_known = not bool(rejected[index])
            records.append(
                {
                    "block_index": int(index),
                    "is_known": is_known,
                    "label": (
                        self.class_names[class_index]
                        if is_known
                        else "UNKNOWN"
                    ),
                    "closest_known_class": self.class_names[class_index],
                    "knownness_score": float(score[index]),
                    "unknown_cluster_id": (
                        int(cluster_ids[index])
                        if cluster_ids[index] >= 0
                        else None
                    ),
                    "class_distances": {
                        name: float(distances[index, class_id])
                        for class_id, name in enumerate(self.class_names)
                    },
                }
            )
        return {
            "model_version": self.version,
            "protocol": "strict_open_set_known_only",
            "known_classes": self.class_names,
            "total_blocks": int(len(records)),
            "known_blocks": int(np.sum(~rejected)),
            "unknown_blocks": int(np.sum(rejected)),
            "known_rate": float(np.mean(~rejected)),
            "unknown_rate": float(np.mean(rejected)),
            "unknown_cluster_count": int(
                len(np.unique(cluster_ids[cluster_ids >= 0]))
            ),
            "cluster_warning": cluster_warning,
            "records": records,
        }

    def predict_file(
        self,
        path: Path,
        hdf5_dataset: Optional[str] = None,
        dat_dtype: str = "complex64",
        cluster_unknown: bool = True,
    ) -> Dict[str, Any]:
        iq = load_iq_file(
            path,
            hdf5_dataset=hdf5_dataset,
            dat_dtype=dat_dtype,
        )
        result = self.predict_array(iq, cluster_unknown=cluster_unknown)
        result["source_file"] = str(Path(path).resolve())
        if hdf5_dataset is not None:
            result["hdf5_dataset"] = str(hdf5_dataset)
        return result

    def evaluate_npz(self, path: Path) -> Dict[str, Any]:
        """评估带 labels 的项目 NPZ，并返回适合软件展示的普通字典。"""
        path = Path(path).resolve()
        with np.load(path, allow_pickle=False) as handle:
            iq = np.asarray(handle["iq"])
            labels = np.asarray(handle["labels"], dtype=np.int64)
            all_names = [str(item) for item in handle["class_names"]]
        result = self.predict_array(iq, cluster_unknown=True)
        actual = np.asarray([all_names[index] for index in labels])
        predicted = np.asarray(
            [record["label"] for record in result["records"]]
        )
        closest = np.asarray(
            [
                record["closest_known_class"]
                for record in result["records"]
            ]
        )
        scores = np.asarray(
            [record["knownness_score"] for record in result["records"]],
            dtype=np.float64,
        )
        cluster_ids = np.asarray(
            [
                -1
                if record["unknown_cluster_id"] is None
                else int(record["unknown_cluster_id"])
                for record in result["records"]
            ],
            dtype=np.int64,
        )
        known_mask = np.isin(actual, self.class_names)
        unknown_mask = ~known_mask
        if not np.any(known_mask) or not np.any(unknown_mask):
            raise ValueError(
                "完整评估 NPZ 必须同时包含已知类和未知类；"
                "纯未知 NPZ 请调用 predict_file()"
            )
        accepted = predicted != "UNKNOWN"
        known_correct = closest[known_mask] == actual[known_mask]
        known_acceptance = float(np.mean(accepted[known_mask]))
        unknown_rejection = float(np.mean(~accepted[unknown_mask]))
        closed_accuracy = float(np.mean(known_correct))
        correct_and_accepted = float(
            np.mean(
                known_correct
                & accepted[known_mask]
            )
        )
        binary_labels = known_mask.astype(np.int64)
        fpr, tpr, thresholds = roc_curve(binary_labels, scores)
        auroc = float(roc_auc_score(binary_labels, scores))
        open_matrix = np.asarray(
            [
                [
                    np.sum(known_mask & accepted),
                    np.sum(known_mask & ~accepted),
                ],
                [
                    np.sum(unknown_mask & accepted),
                    np.sum(unknown_mask & ~accepted),
                ],
            ],
            dtype=np.int64,
        )
        unknown_names = [
            name for name in all_names if name not in self.class_names
        ]
        # 正式聚类指标只在“真实未知且已被拒识”的测试样本上计算。
        # 这里的 known/unknown 身份来自测试标签，只用于测试阶段筛选；
        # GMM 拟合仍不读取 AM/FM 类别标签。实际无标签推理继续使用上方
        # predict_array() 产生的 operational cluster_ids。
        evaluation_rejected = unknown_mask & ~accepted
        clustering: Dict[str, Any] = {
            "sample_count": int(np.sum(evaluation_rejected)),
            "nmi": None,
            "ari": None,
            "purity": None,
            "cluster_accuracy": None,
            "evaluation_protocol": (
                "cluster true-unknown rejected test samples; "
                "AM/FM labels are not used for GMM fitting"
            ),
        }
        if np.sum(evaluation_rejected) >= 2 and len(unknown_names) >= 2:
            evaluation_blocks = _fit_block_length(
                _as_iq_channels(iq), self.sample_count
            )[evaluation_rejected]
            evaluation_features = analog_modulation_features(
                evaluation_blocks
            )
            evaluation_scaled = StandardScaler().fit_transform(
                evaluation_features
            )
            evaluation_clusters = GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=1e-4,
                n_init=30,
                random_state=20260806,
            ).fit_predict(evaluation_scaled)
            true_cluster = np.asarray(
                [
                    unknown_names.index(name)
                    for name in actual[evaluation_rejected]
                ],
                dtype=np.int64,
            )
            clustering.update(
                clustering_metrics(true_cluster, evaluation_clusters)
            )
        display_metrics = {
            "known_signal_recognition_rate": known_acceptance,
            "known_signal_recognition_rate_definition": (
                "真实已知信号被判断为已知的比例（已知接受率）"
            ),
            "unknown_rejection_rate": unknown_rejection,
            "auroc": auroc,
            "unknown_clustering_nmi": clustering["nmi"],
            "unknown_clustering_accuracy": clustering[
                "cluster_accuracy"
            ],
        }
        diagnostic_metrics = {
            "known_closed_set_four_class_accuracy": closed_accuracy,
            "known_correct_and_accepted_rate": correct_and_accepted,
            "open_binary_balanced_accuracy": float(
                0.5 * (known_acceptance + unknown_rejection)
            ),
            "oscr": _oscr_scalar(
                scores[known_mask],
                scores[unknown_mask],
                known_correct,
            ),
            "known_sample_count": int(np.sum(known_mask)),
            "unknown_sample_count": int(np.sum(unknown_mask)),
            "per_unknown_class_rejection_rate": {
                name: float(
                    np.mean(~accepted[actual == name])
                )
                for name in unknown_names
                if np.any(actual == name)
            },
        }
        result.update(
            {
                "source_file": str(path),
                "display_metrics": display_metrics,
                "diagnostic_metrics": diagnostic_metrics,
                "unknown_clustering": clustering,
                "open_set_confusion_matrix": open_matrix.tolist(),
                "open_set_confusion_matrix_labels": {
                    "rows_actual": ["KNOWN", "UNKNOWN"],
                    "columns_predicted": ["KNOWN", "UNKNOWN"],
                },
                "roc_curve": {
                    "false_positive_rate": fpr.tolist(),
                    "true_positive_rate": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                },
            }
        )
        return result
