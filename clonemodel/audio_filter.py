"""Audio noise filter utilities."""
import numpy as np

def reduce_background_noise(audio_samples: np.ndarray) -> np.ndarray:
    if audio_samples is None or len(audio_samples) == 0:
        return audio_samples
    threshold = 0.005
    audio_samples = np.where(np.abs(audio_samples) < threshold, 0.0, audio_samples)
    return audio_samples


def normalize_audio_volume(audio_samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    if audio_samples is None or len(audio_samples) == 0:
        return audio_samples
    max_val = np.max(np.abs(audio_samples))
    if max_val > 0:
        audio_samples = audio_samples * (target_peak / max_val)
    return audio_samples
