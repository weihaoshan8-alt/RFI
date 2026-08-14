"""低信噪比数字调制的循环谱特征。

利用 BPSK 的二次幂循环谱线与 16QAM 的非圆星座差异。所有特征均从原始 IQ
直接计算，对未知载波频移天然不敏感；不读取类别标签。
"""

from __future__ import annotations

from typing import List

import numpy as np


def _spectral_summary(values: np.ndarray) -> List[np.ndarray]:
    power = np.abs(np.fft.fft(values, axis=1)) ** 2
    median = np.median(power, axis=1)
    mean = np.mean(power, axis=1)
    peak_index = np.argmax(power, axis=1)
    rows = np.arange(len(values))
    local = np.zeros(len(values), dtype=np.float64)
    for shift in range(-3, 4):
        local += power[rows, (peak_index + shift) % power.shape[1]]
    ordered = np.partition(power, -4, axis=1)[:, -4:]
    entropy_source = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-12)
    return [
        np.max(power, axis=1) / np.maximum(median, 1e-12),
        local / np.maximum(median, 1e-12),
        ordered[:, -2] / np.maximum(median, 1e-12),
        ordered[:, -3] / np.maximum(median, 1e-12),
        np.quantile(power, 0.95, axis=1) / np.maximum(median, 1e-12),
        np.quantile(power, 0.99, axis=1) / np.maximum(median, 1e-12),
        np.quantile(power, 0.999, axis=1) / np.maximum(median, 1e-12),
        np.mean(power ** 2, axis=1) / np.maximum(mean ** 2, 1e-12),
        -np.sum(entropy_source * np.log(np.maximum(entropy_source, 1e-15)), axis=1),
    ]


def cyclic_spectral_features(iq: np.ndarray) -> np.ndarray:
    """提取多阶循环谱线、幅度矩和延迟循环相关特征。"""
    values = np.asarray(iq, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("IQ 必须是 [B,2,N]")
    signal = values[:, 0] + 1j * values[:, 1]
    signal = signal / np.sqrt(np.maximum(np.mean(np.abs(signal) ** 2, axis=1, keepdims=True), 1e-12))
    rows: List[np.ndarray] = []
    amplitude = np.abs(signal)
    for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        rows.append(np.quantile(amplitude, quantile, axis=1))
    for order in (1, 2, 3, 4):
        transformed = signal ** order
        rows.extend(_spectral_summary(transformed))
        for lag in (1, 2, 4, 8, 16, 32):
            correlation = np.mean(
                transformed[:, lag:] * np.conj(transformed[:, :-lag]), axis=1
            )
            rows.append(np.abs(correlation))
    return np.nan_to_num(np.stack(rows, axis=1), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
