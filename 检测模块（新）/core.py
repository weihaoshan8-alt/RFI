"""工程门限能量检测核心算法。

统计量与老师给出的 MATLAB 方法一致：

    T = sum(I[n] ** 2 + Q[n] ** 2)

当 T >= threshold 时，判定当前 IQ 数据块中存在信号。
门限标定算法已独立放入 threshold_calibration.py；本文件仅保留检测判决。
为兼容旧调用方式，calculate_block_energies 和 empirical_threshold 仍可从本文件导入。
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np

from threshold_calibration import calculate_block_energies, empirical_threshold


SIGNAL_TYPES = ("AM", "FM", "2ASK", "2FSK", "BPSK", "16QAM")


@dataclass(frozen=True)
class DetectionResult:
    """单个 IQ 数据块的检测结果。"""

    has_signal: bool
    energy: float
    threshold: float
    energy_margin: float
    energy_ratio: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnergyDetector:
    """可供软件界面直接调用的能量检测器。"""

    def __init__(self, threshold: float, sample_count: int):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold 必须是大于 0 的有限值")
        if sample_count <= 0:
            raise ValueError("sample_count 必须是正整数")
        self.threshold = float(threshold)
        self.sample_count = int(sample_count)

    @classmethod
    def from_noise_energies(
        cls,
        noise_energies: np.ndarray,
        sample_count: int,
        pfa_target: float = 0.05,
    ) -> Tuple["EnergyDetector", int]:
        threshold, rank = empirical_threshold(noise_energies, pfa_target)
        return cls(threshold, sample_count), rank

    def detect_batch(self, iq_data: np.ndarray) -> np.ndarray:
        """批量检测，返回一维 bool 数组；True 表示检测到信号。"""
        iq = np.asarray(iq_data)
        if iq.ndim == 3:
            actual_sample_count = iq.shape[2] if iq.shape[1] == 2 else iq.shape[1]
        elif iq.ndim == 2:
            if np.iscomplexobj(iq):
                actual_sample_count = iq.shape[1]
            else:
                actual_sample_count = iq.shape[1] if iq.shape[0] == 2 else iq.shape[0]
        elif iq.ndim == 1 and np.iscomplexobj(iq):
            actual_sample_count = iq.shape[0]
        else:
            raise ValueError("无法确定 IQ 数据块长度")

        if int(actual_sample_count) != self.sample_count:
            raise ValueError(
                "IQ 块长度不匹配：要求 {} 点，实际 {} 点".format(
                    self.sample_count, actual_sample_count
                )
            )
        energies = calculate_block_energies(iq)
        return energies >= self.threshold

    def detect(self, iq_block: np.ndarray) -> DetectionResult:
        """检测一个 IQ 数据块并返回详细结果。"""
        energies = calculate_block_energies(iq_block)
        if energies.size != 1:
            raise ValueError("detect() 只接受一个 IQ 数据块")
        energy = float(energies[0])
        return DetectionResult(
            has_signal=energy >= self.threshold,
            energy=energy,
            threshold=self.threshold,
            energy_margin=energy - self.threshold,
            energy_ratio=energy / self.threshold,
            sample_count=self.sample_count,
        )
