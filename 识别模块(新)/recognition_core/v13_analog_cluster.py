"""Generic narrowband amplitude/frequency modulation clustering features."""

from typing import List

import numpy as np
from scipy.ndimage import uniform_filter1d


def _spectral_tone_metrics(values: np.ndarray):
    centered = values - np.mean(values, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    spectrum[:, 0] = 0.0
    total = np.sum(spectrum, axis=1)
    peak = np.max(spectrum, axis=1) / np.maximum(total, 1e-12)
    probability = spectrum / np.maximum(total[:, None], 1e-12)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1e-15)), axis=1
    ) / np.log(spectrum.shape[1])
    return peak, entropy


def analog_modulation_features(iq: np.ndarray) -> np.ndarray:
    """Extract phase-invariant AM/FM descriptors after carrier alignment."""

    values = np.asarray(iq, dtype=np.float64)
    batch, _, sample_count = values.shape
    signal = values[:, 0] + 1j * values[:, 1]
    spectrum = np.fft.fftshift(np.fft.fft(signal, axis=1), axes=1)
    power = np.abs(spectrum) ** 2
    smooth = uniform_filter1d(
        power,
        size=min(256, sample_count // 16),
        axis=1,
        mode="wrap",
    )
    detected = np.argmax(smooth, axis=1)
    target = sample_count // 2
    aligned = np.stack(
        [
            np.roll(spectrum[index], target - int(detected[index]))
            for index in range(batch)
        ],
        axis=0,
    )
    rows: List[np.ndarray] = []
    for width in (128, 256, 512, 1024):
        half = width // 2
        masked = np.zeros_like(aligned)
        masked[:, target - half : target + half] = aligned[
            :, target - half : target + half
        ]
        narrow = np.fft.ifft(
            np.fft.ifftshift(masked, axes=1), axis=1
        )
        narrow /= np.sqrt(
            np.maximum(
                np.mean(np.abs(narrow) ** 2, axis=1, keepdims=True),
                1e-12,
            )
        )
        envelope = np.abs(narrow)
        phase_step = np.angle(narrow[:, 1:] * np.conj(narrow[:, :-1]))
        envelope_peak, envelope_entropy = _spectral_tone_metrics(envelope)
        frequency_peak, frequency_entropy = _spectral_tone_metrics(phase_step)
        envelope_std = np.std(envelope, axis=1)
        phase_std = np.std(phase_step, axis=1)
        envelope_span = (
            np.quantile(envelope, 0.95, axis=1)
            - np.quantile(envelope, 0.05, axis=1)
        )
        frequency_span = (
            np.quantile(np.abs(phase_step), 0.95, axis=1)
            - np.quantile(np.abs(phase_step), 0.05, axis=1)
        )
        rows.append(
            np.stack(
                [
                    envelope_std,
                    phase_std,
                    envelope_span,
                    frequency_span,
                    envelope_peak,
                    frequency_peak,
                    envelope_entropy,
                    frequency_entropy,
                    np.log(
                        np.maximum(envelope_peak, 1e-12)
                        / np.maximum(frequency_peak, 1e-12)
                    ),
                    envelope_std / np.maximum(phase_std, 1e-12),
                ],
                axis=1,
            )
        )
    output = np.concatenate(rows, axis=1)
    return np.nan_to_num(
        output, nan=0.0, posinf=1e6, neginf=-1e6
    ).astype(np.float32)
