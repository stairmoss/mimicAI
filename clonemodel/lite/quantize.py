import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizedTensor:
    data: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    shape: torch.Size
    bits: int
    group_size: int


def quantize_tensor(tensor: torch.Tensor, bits: int = 4, group_size: int = 128) -> QuantizedTensor:
    assert bits in (4, 8)
    original_shape = tensor.shape
    flat = tensor.flatten().float()
    n = flat.numel()

    if n % group_size != 0:
        pad_len = group_size - (n % group_size)
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=flat.dtype)])
    else:
        pad_len = 0

    num_groups = flat.numel() // group_size
    grouped = flat.view(num_groups, group_size)

    g_min = grouped.min(dim=1, keepdim=True).values
    g_max = grouped.max(dim=1, keepdim=True).values
    qmax = (1 << bits) - 1

    scale = (g_max - g_min) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = g_min

    quantized = torch.clamp(
        torch.round((grouped - zero) / scale), 0, qmax
    ).to(torch.uint8)

    quantized_flat = quantized.flatten()
    if pad_len > 0:
        quantized_flat = quantized_flat[:n]

    if bits == 4:
        packed = _pack_int4(quantized_flat)
    else:
        packed = quantized_flat.to(torch.uint8)

    return QuantizedTensor(
        data=packed, scales=scale.squeeze(1), zeros=zero.squeeze(1),
        shape=original_shape, bits=bits, group_size=group_size,
    )


def dequantize_tensor(qt: QuantizedTensor) -> torch.Tensor:
    n = 1
    for s in qt.shape:
        n *= s

    if qt.bits == 4:
        quantized_flat = _unpack_int4(qt.data, n)
    else:
        quantized_flat = qt.data[:n].to(torch.float32)

    group_size = qt.group_size
    if n % group_size != 0:
        pad_len = group_size - (n % group_size)
        quantized_flat = torch.cat([quantized_flat, torch.zeros(pad_len, dtype=quantized_flat.dtype)])
    else:
        pad_len = 0

    num_groups = quantized_flat.numel() // group_size
    grouped = quantized_flat.view(num_groups, group_size).float()

    scales = qt.scales[:num_groups].unsqueeze(1)
    zeros = qt.zeros[:num_groups].unsqueeze(1)
    dequantized = grouped * scales + zeros

    result = dequantized.flatten()[:n]
    return result.view(qt.shape)


def _pack_int4(data: torch.Tensor) -> torch.Tensor:
    n = data.numel()
    if n % 2 != 0:
        data = torch.cat([data, torch.zeros(1, dtype=data.dtype)])
    even = data[0::2].to(torch.uint8)
    odd = data[1::2].to(torch.uint8)
    packed = (odd << 4) | (even & 0x0F)
    return packed


def _unpack_int4(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.float32)
    high = ((packed >> 4) & 0x0F).to(torch.float32)
    result = torch.stack([low, high], dim=1).flatten()
    return result[:num_elements]


def save_quantized_layer(state_dict: Dict[str, torch.Tensor], output_path: Path, bits: int = 4, group_size: int = 128) -> dict:
    quantized_state = {}
    metadata = {"params": {}, "bits": bits, "group_size": group_size}

    for name, tensor in state_dict.items():
        if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
            qt = quantize_tensor(tensor.float(), bits=bits, group_size=group_size)
            quantized_state[f"{name}.data"] = qt.data
            quantized_state[f"{name}.scales"] = qt.scales
            quantized_state[f"{name}.zeros"] = qt.zeros
            metadata["params"][name] = {"shape": list(qt.shape), "dtype": str(tensor.dtype)}
        else:
            quantized_state[name] = tensor
            metadata["params"][name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "raw": True}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": quantized_state, "metadata": metadata}, output_path)
    return metadata


def load_quantized_layer(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    quantized_state = checkpoint["weights"]
    metadata = checkpoint["metadata"]
    bits = metadata["bits"]
    group_size = metadata["group_size"]

    result = {}
    for name, param_meta in metadata["params"].items():
        if param_meta.get("raw", False):
            result[name] = quantized_state[name].to(device)
            continue

        qt = QuantizedTensor(
            data=quantized_state[f"{name}.data"],
            scales=quantized_state[f"{name}.scales"],
            zeros=quantized_state[f"{name}.zeros"],
            shape=torch.Size(param_meta["shape"]),
            bits=bits, group_size=group_size,
        )
        result[name] = dequantize_tensor(qt).to(device)

        orig_dtype = param_meta["dtype"]
        if "float16" in orig_dtype:
            result[name] = result[name].half()
        elif "bfloat16" in orig_dtype:
            result[name] = result[name].bfloat16()

    return result


def estimate_quantized_size(state_dict: Dict[str, torch.Tensor], bits: int = 4, group_size: int = 128) -> dict:
    original_bytes = 0
    quantized_bytes = 0

    for name, tensor in state_dict.items():
        param_bytes = tensor.numel() * tensor.element_size()
        original_bytes += param_bytes

        if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
            n = tensor.numel()
            if bits == 4:
                data_bytes = (n + 1) // 2
            else:
                data_bytes = n
            num_groups = (n + group_size - 1) // group_size
            meta_bytes = num_groups * 4 * 2
            quantized_bytes += data_bytes + meta_bytes
        else:
            quantized_bytes += param_bytes

    return {
        "original_mb": original_bytes / (1024 * 1024),
        "quantized_mb": quantized_bytes / (1024 * 1024),
        "compression_ratio": original_bytes / max(quantized_bytes, 1),
    }
