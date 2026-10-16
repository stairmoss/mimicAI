#!/usr/bin/env python3

import logging
from typing import Optional

import soundfile as sf
import torch
import torchaudio


def load_eval_waveform(
    fname: str,
    sample_rate: int,
    dtype: str = "float32",
    device: torch.device = torch.device("cpu"),
    return_numpy: bool = False,
    max_seconds: Optional[float] = None,
) -> torch.Tensor:
    """
    Load an audio file, preprocess it, and convert to a PyTorch tensor.

    Args:
        fname (str): Path to the audio file.
        sample_rate (int): Target sample rate for resampling.
        dtype (str, optional): Data type to load audio as (default: "float32").
        device (torch.device, optional): Device to place the resulting tensor
            on (default: CPU).
        return_numpy (bool): If True, returns a NumPy array instead of a
            PyTorch tensor.
        max_seconds (float): Maximum length (seconds) of the audio tensor.
            If the audio is longer than this, it will be truncated.

    Returns:
        torch.Tensor: Processed audio waveform as a PyTorch tensor,
            with shape (num_samples,).

    Notes:
        - If the audio is stereo, it will be converted to mono by averaging channels.
        - If the audio's sample rate differs from the target, it will be resampled.
    """
    wav_data, sr = sf.read(fname, dtype=dtype)

    if len(wav_data.shape) == 2:
        wav_data = wav_data.mean(1)

    if sr != sample_rate:
        wav_data = torchaudio.functional.resample(
            torch.from_numpy(wav_data), orig_freq=sr, new_freq=sample_rate
        ).numpy()

    if max_seconds is not None:
        max_length = int(sample_rate * max_seconds)
        if len(wav_data) > max_length:
            wav_data = wav_data[:max_length]
            logging.warning(
                f"Wav file {fname} is longer than {max_seconds}s, "
                f"truncated to {max_seconds}s to avoid OOM."
            )
    if return_numpy:
        return wav_data
    else:
        wav_data = torch.from_numpy(wav_data)
        return wav_data.to(device)
