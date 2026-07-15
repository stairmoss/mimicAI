"""Audio noise filter utilities."""
import numpy as np

def reduce_background_noise(audio_samples: np.ndarray) -> np.ndarray:
    if audio_samples is None or len(audio_samples) == 0:
        return audio_samples
    threshold = 0.005
    audio_samples = np.where(np.abs(audio_samples) < threshold, 0.0, audio_samples)
    return audio_samples
