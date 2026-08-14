"""六类复基带IQ信号的批量生成。

数字调制链路沿用师兄代码中的基本口径：
随机符号 -> 上采样 -> RRC成形（2FSK使用连续相位频率调制）。
AM/FM按测试大纲的10--100 kHz带宽范围生成。
"""

import math
from typing import Tuple

import numpy as np
from scipy.signal import lfilter


ALL_CLASSES = ("AM", "FM", "2ASK", "2FSK", "BPSK", "16QAM")


def root_raised_cosine(rolloff: float, samples_per_symbol: int, span: int) -> np.ndarray:
    """生成单位能量RRC脉冲，span单位为码元。"""
    half = span * samples_per_symbol
    t = np.arange(-half, half + 1, dtype=np.float64) / samples_per_symbol
    taps = np.empty_like(t)
    beta = float(rolloff)
    for index, value in enumerate(t):
        if abs(value) < 1e-12:
            taps[index] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(value) - 1.0 / (4.0 * beta)) < 1e-12:
            taps[index] = (
                beta
                / math.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * math.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * math.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            numerator = (
                math.sin(np.pi * value * (1.0 - beta))
                + 4.0 * beta * value * math.cos(np.pi * value * (1.0 + beta))
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[index] = numerator / denominator
    taps /= math.sqrt(float(np.sum(taps ** 2)))
    return taps


def _normalise_each(blocks: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(blocks) ** 2, axis=1, keepdims=True)
    if np.any(power <= 0.0):
        raise ValueError("生成了零功率信号")
    return blocks / np.sqrt(power)


def _apply_carrier_variation(
    blocks: np.ndarray,
    rng: np.random.Generator,
    sample_rate_hz: float,
    frequency_offset_max_hz: float,
) -> np.ndarray:
    block_count, sample_count = blocks.shape
    offsets = rng.uniform(
        -frequency_offset_max_hz, frequency_offset_max_hz, size=(block_count, 1)
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(block_count, 1))
    time = np.arange(sample_count, dtype=np.float64)[None, :] / sample_rate_hz
    return blocks * np.exp(1j * (2.0 * np.pi * offsets * time + phases))


def _digital_symbols(
    signal_type: str, rng: np.random.Generator, shape: Tuple[int, int]
) -> np.ndarray:
    if signal_type == "2ASK":
        return rng.integers(0, 2, size=shape).astype(np.float64).astype(np.complex128)
    if signal_type == "BPSK":
        bits = rng.integers(0, 2, size=shape)
        return (1.0 - 2.0 * bits).astype(np.complex128)
    if signal_type == "16QAM":
        i_index = rng.integers(0, 4, size=shape)
        q_index = rng.integers(0, 4, size=shape)
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        symbols = levels[i_index] + 1j * levels[q_index]
        return symbols / math.sqrt(10.0)
    raise ValueError("不支持的成形数字信号: {}".format(signal_type))


def _generate_shaped_digital(
    signal_type: str,
    rng: np.random.Generator,
    block_count: int,
    sample_count: int,
    sample_rate_hz: float,
    symbol_rate_hz: float,
) -> np.ndarray:
    samples_per_symbol = int(round(sample_rate_hz / symbol_rate_hz))
    taps = root_raised_cosine(0.5, samples_per_symbol, span=4)
    margin = len(taps) + samples_per_symbol * 2
    output_points = sample_count + margin
    symbol_count = int(math.ceil(output_points / samples_per_symbol)) + 2
    symbols = _digital_symbols(signal_type, rng, (block_count, symbol_count))
    upsampled = np.zeros(
        (block_count, symbol_count * samples_per_symbol), dtype=np.complex128
    )
    upsampled[:, ::samples_per_symbol] = symbols
    shaped = lfilter(taps, [1.0], upsampled, axis=1)
    start = len(taps) // 2 + samples_per_symbol
    return shaped[:, start : start + sample_count]


def _generate_2fsk(
    rng: np.random.Generator,
    block_count: int,
    sample_count: int,
    sample_rate_hz: float,
    symbol_rate_hz: float,
) -> np.ndarray:
    samples_per_symbol = int(round(sample_rate_hz / symbol_rate_hz))
    symbol_count = int(math.ceil(sample_count / samples_per_symbol)) + 2
    symbols = rng.integers(0, 2, size=(block_count, symbol_count))
    symbol_frequency = np.where(symbols == 0, -symbol_rate_hz, symbol_rate_hz)
    instantaneous_frequency = np.repeat(symbol_frequency, samples_per_symbol, axis=1)
    instantaneous_frequency = instantaneous_frequency[:, :sample_count]
    initial_phase = rng.uniform(0.0, 2.0 * np.pi, size=(block_count, 1))
    phase = initial_phase + 2.0 * np.pi * np.cumsum(
        instantaneous_frequency / sample_rate_hz, axis=1
    )
    return np.exp(1j * phase)


def _generate_am(
    rng: np.random.Generator,
    block_count: int,
    sample_count: int,
    sample_rate_hz: float,
    bandwidth_min_hz: float,
    bandwidth_max_hz: float,
) -> np.ndarray:
    bandwidth = rng.uniform(bandwidth_min_hz, bandwidth_max_hz, size=(block_count, 1))
    message_frequency = bandwidth / 2.0
    message_phase = rng.uniform(0.0, 2.0 * np.pi, size=(block_count, 1))
    modulation_index = rng.uniform(0.4, 0.9, size=(block_count, 1))
    time = np.arange(sample_count, dtype=np.float64)[None, :] / sample_rate_hz
    return 1.0 + modulation_index * np.cos(
        2.0 * np.pi * message_frequency * time + message_phase
    )


def _generate_fm(
    rng: np.random.Generator,
    block_count: int,
    sample_count: int,
    sample_rate_hz: float,
    bandwidth_min_hz: float,
    bandwidth_max_hz: float,
) -> np.ndarray:
    bandwidth = rng.uniform(bandwidth_min_hz, bandwidth_max_hz, size=(block_count, 1))
    message_frequency = bandwidth / 10.0
    frequency_deviation = np.maximum(bandwidth / 2.0 - message_frequency, 1.0)
    modulation_index = frequency_deviation / message_frequency
    message_phase = rng.uniform(0.0, 2.0 * np.pi, size=(block_count, 1))
    carrier_phase = rng.uniform(0.0, 2.0 * np.pi, size=(block_count, 1))
    time = np.arange(sample_count, dtype=np.float64)[None, :] / sample_rate_hz
    phase = carrier_phase + modulation_index * np.sin(
        2.0 * np.pi * message_frequency * time + message_phase
    )
    return np.exp(1j * phase)


def generate_clean_batch(
    signal_type: str,
    rng: np.random.Generator,
    block_count: int,
    sample_count: int,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    bandwidth_min_hz: float,
    bandwidth_max_hz: float,
    frequency_offset_max_hz: float,
) -> np.ndarray:
    if signal_type not in ALL_CLASSES:
        raise ValueError("不支持的信号类型: {}".format(signal_type))
    if signal_type == "AM":
        blocks = _generate_am(
            rng,
            block_count,
            sample_count,
            sample_rate_hz,
            bandwidth_min_hz,
            bandwidth_max_hz,
        )
    elif signal_type == "FM":
        blocks = _generate_fm(
            rng,
            block_count,
            sample_count,
            sample_rate_hz,
            bandwidth_min_hz,
            bandwidth_max_hz,
        )
    elif signal_type == "2FSK":
        blocks = _generate_2fsk(
            rng, block_count, sample_count, sample_rate_hz, symbol_rate_hz
        )
    else:
        blocks = _generate_shaped_digital(
            signal_type,
            rng,
            block_count,
            sample_count,
            sample_rate_hz,
            symbol_rate_hz,
        )
    blocks = _normalise_each(np.asarray(blocks, dtype=np.complex128))
    blocks = _apply_carrier_variation(
        blocks, rng, sample_rate_hz, frequency_offset_max_hz
    )
    return _normalise_each(blocks)


def add_complex_awgn(
    clean_blocks: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
    noise_power: float,
) -> np.ndarray:
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    scaled_signal = clean_blocks * math.sqrt(noise_power * snr_linear)
    scale = math.sqrt(noise_power / 2.0)
    noise = scale * (
        rng.standard_normal(clean_blocks.shape)
        + 1j * rng.standard_normal(clean_blocks.shape)
    )
    return scaled_signal + noise


def to_iq_channels(complex_blocks: np.ndarray) -> np.ndarray:
    output = np.empty(
        (complex_blocks.shape[0], 2, complex_blocks.shape[1]), dtype=np.float32
    )
    output[:, 0, :] = np.real(complex_blocks)
    output[:, 1, :] = np.imag(complex_blocks)
    return output

