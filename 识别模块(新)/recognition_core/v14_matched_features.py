"""面向低信噪比数字调制闭集识别的匹配滤波幅度特征。

特征只使用四类已知 IQ 数据。其核心思想是先按已知生成参数进行 RRC 匹配滤波，
再汇总八个候选符号定时相位的归一化幅度分布。这样即使存在未知载波相位和
频偏，BPSK 的近恒包络特性和 16QAM 的多幅度层级仍可保留给闭集分类器。
"""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.signal import lfilter

from .signals import root_raised_cosine


_SPS = 8
_TAPS = root_raised_cosine(0.5, samples_per_symbol=_SPS, span=4)
_HIST_EDGES = np.linspace(0.0, 4.0, 21, dtype=np.float64)


def _phase_features(symbols: np.ndarray) -> np.ndarray:
    """计算一个候选定时相位的固定长度幅度统计特征。"""
    power = np.abs(symbols) ** 2
    norm_power = power / np.maximum(np.mean(power, axis=1, keepdims=True), 1e-12)
    amplitude = np.sqrt(norm_power)
    centered = norm_power - 1.0
    columns: List[np.ndarray] = [
        np.mean(amplitude, axis=1),
        np.std(amplitude, axis=1),
        np.mean(centered ** 2, axis=1),
        np.mean(centered ** 3, axis=1),
        np.mean(centered ** 4, axis=1),
        np.mean(np.abs(centered), axis=1),
    ]
    for quantile in (0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
        columns.append(np.quantile(amplitude, quantile, axis=1))
    histograms = []
    for row in amplitude:
        histograms.append(np.histogram(row, bins=_HIST_EDGES, density=True)[0])
    columns.extend(np.asarray(histograms, dtype=np.float64).T)
    for lag in (1, 2, 3, 4, 8, 16):
        correlation = np.mean(
            centered[:, lag:] * centered[:, :-lag], axis=1
        )
        columns.append(correlation)
    return np.stack(columns, axis=1)


def matched_amplitude_features(iq: np.ndarray) -> np.ndarray:
    """由 ``[B,2,N]`` IQ 块提取 RRC 匹配滤波后的稳健闭集特征。"""
    values = np.asarray(iq, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("IQ 必须是 [B,2,N]")
    signal = values[:, 0] + 1j * values[:, 1]
    matched = lfilter(_TAPS, [1.0], signal, axis=1)
    trim = len(_TAPS)
    if matched.shape[1] > 2 * trim + _SPS:
        matched = matched[:, trim:-trim]
    phase_values = []
    for phase in range(_SPS):
        phase_values.append(_phase_features(matched[:, phase::_SPS]))
    phase_tensor = np.stack(phase_values, axis=1)
    # 对每个统计量汇总八个未知符号定时相位，避免将固定定时位置作为捷径。
    summaries = [
        np.mean(phase_tensor, axis=1),
        np.std(phase_tensor, axis=1),
        np.min(phase_tensor, axis=1),
        np.max(phase_tensor, axis=1),
    ]
    # 追加幅度起伏最小的候选相位，常接近真实符号抽样位置。
    best_phase = np.argmin(phase_tensor[:, :, 1], axis=1)
    summaries.append(phase_tensor[np.arange(len(values)), best_phase])
    output = np.concatenate(summaries, axis=1)
    return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def cfo_matched_amplitude_features(iq: np.ndarray) -> np.ndarray:
    """先用相邻采样自相关估计载频偏移，再提取匹配滤波幅度特征。

    对本项目的 RRC 成形数字信号，单采样间隔自相关在无频偏时近似为正实数；
    其相角可以估计每采样点的载频旋转。该量从 IQ 本身计算，不依赖类别标签。
    """
    values = np.asarray(iq, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("IQ 必须是 [B,2,N]")
    signal = values[:, 0] + 1j * values[:, 1]
    adjacent = np.mean(signal[:, 1:] * np.conj(signal[:, :-1]), axis=1)
    phase_per_sample = np.angle(adjacent)
    time = np.arange(signal.shape[1], dtype=np.float64)[None, :]
    corrected = signal * np.exp(-1j * phase_per_sample[:, None] * time)
    corrected_iq = np.stack((corrected.real, corrected.imag), axis=1)
    return matched_amplitude_features(corrected_iq)
