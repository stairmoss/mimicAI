import gc
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clonemodel.lite.memory_manager import TieredMemoryManager
from clonemodel.lite.quantize import load_quantized_layer

logger = logging.getLogger(__name__)


class LightweightTransformerLayer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_heads = config.get("num_attention_heads", 16)
        self.num_kv_heads = config.get("num_key_value_heads", 8)
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size = config.get("intermediate_size", 3072)
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        self._weights: Dict[str, torch.Tensor] = {}

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        self._weights = state_dict

    def clear_weights(self):
        self._weights.clear()
        gc.collect()

    def _get(self, key: str) -> torch.Tensor:
        if key in self._weights:
            return self._weights[key]
        variants = [key, key.replace("self_attn.", ""), key.replace("mlp.", ""),
                    f"self_attn.{key}", f"mlp.{key}", f"input_layernorm.{key}", f"post_attention_layernorm.{key}"]
        for v in variants:
            if v in self._weights:
                return self._weights[v]
        available = list(self._weights.keys())
        raise KeyError(f"Weight '{key}' not found. Available: {available[:20]}")

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "input_layernorm.weight")
        attn_output, new_kv = self._attention(hidden_states, attention_mask, position_ids, past_key_value)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "post_attention_layernorm.weight")
        hidden_states = self._mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_kv

    def _rms_norm(self, x, weight_key):
        weight = self._get(weight_key).to(x.dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x * weight

    def _attention(self, hidden_states, attention_mask, position_ids, past_key_value):
        bsz, q_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        q_weight = self._get("self_attn.q_proj.weight").to(dtype)
        k_weight = self._get("self_attn.k_proj.weight").to(dtype)
        v_weight = self._get("self_attn.v_proj.weight").to(dtype)
        o_weight = self._get("self_attn.o_proj.weight").to(dtype)

        try:
            q_bias = self._get("self_attn.q_proj.bias").to(dtype)
        except KeyError:
            q_bias = None
        try:
            k_bias = self._get("self_attn.k_proj.bias").to(dtype)
        except KeyError:
            k_bias = None
        try:
            v_bias = self._get("self_attn.v_proj.bias").to(dtype)
        except KeyError:
            v_bias = None

        query = F.linear(hidden_states, q_weight, q_bias)
        key = F.linear(hidden_states, k_weight, k_bias)
        value = F.linear(hidden_states, v_weight, v_bias)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_ids is not None:
            cos, sin = self._get_rotary_embedding(q_len, position_ids, dtype)
            query = self._apply_rotary_pos_emb(query, cos, sin)
            key = self._apply_rotary_pos_emb(key, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key = torch.cat([past_k.to(key.device), key], dim=2)
            value = torch.cat([past_v.to(value.device), value], dim=2)

        new_kv = (key.cpu(), value.cpu())

        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = F.linear(attn_output, o_weight)
        return attn_output, new_kv

    def _mlp(self, hidden_states):
        dtype = hidden_states.dtype
        gate_weight = self._get("mlp.gate_proj.weight").to(dtype)
        up_weight = self._get("mlp.up_proj.weight").to(dtype)
        down_weight = self._get("mlp.down_proj.weight").to(dtype)
        gate = F.linear(hidden_states, gate_weight)
        up = F.linear(hidden_states, up_weight)
        hidden_states = F.silu(gate) * up
        hidden_states = F.linear(hidden_states, down_weight)
        return hidden_states

    def _get_rotary_embedding(self, seq_len, position_ids, dtype):
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=position_ids.device).float() / self.head_dim))
        pos = position_ids.unsqueeze(-1).float()
        freqs = pos * inv_freq.unsqueeze(0).unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(1).to(dtype)
        sin = emb.sin().unsqueeze(1).to(dtype)
        return cos, sin

    def _apply_rotary_pos_emb(self, x, cos, sin):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        seq_dim = 2
        cos = cos[:, :, :x.shape[seq_dim], :]
        sin = sin[:, :, :x.shape[seq_dim], :]
        return x * cos + rotated * sin


class LayerWiseForwardPass:
    def __init__(self, memory_manager, config, progress_callback=None):
        self.mm = memory_manager
        self.config = config
        self.progress_callback = progress_callback
        self._layer = LightweightTransformerLayer(config)

    def run_forward(self, inputs_embeds, attention_mask=None, position_ids=None, num_layers=28, prefix="llm_layer", use_kv_cache=False):
        hidden_states = inputs_embeds
        total_start = time.monotonic()

        if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        for layer_idx in range(num_layers):
            layer_id = f"{prefix}_{layer_idx:02d}"

            if layer_idx + 1 < num_layers:
                self.mm.prefetch_layer(f"{prefix}_{layer_idx + 1:02d}")

            weights = self.mm.load_layer(layer_id)
            self._layer.load_weights(weights)

            past_kv = None
            if use_kv_cache:
                kv = self.mm.fetch_kv_cache(layer_idx)
                if kv is not None:
                    past_kv = (kv["key"], kv["value"])

            hidden_states, new_kv = self._layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_kv)

            if use_kv_cache and new_kv is not None:
                self.mm.store_kv_cache(layer_idx, {"key": new_kv[0], "value": new_kv[1]})

            self._layer.clear_weights()
            self.mm.evict_layer(layer_id)

            if self.progress_callback:
                self.progress_callback(layer_idx + 1, num_layers, f"Layer {layer_idx}/{num_layers-1}")

        total_elapsed = time.monotonic() - total_start
        logger.info(f"Forward pass ({num_layers} layers) completed in {total_elapsed:.1f}s")
        return hidden_states

    def apply_final_norm(self, hidden_states, norm_weights, eps=1e-6):
        weight = norm_weights.get("weight", None)
        if weight is None:
            return hidden_states
        weight = weight.to(hidden_states.dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return hidden_states * weight


class LayerWiseAudioTokenizer:
    def __init__(self, memory_manager, config, sampling_rate=24000):
        self.mm = memory_manager
        self.config = config
        self.sampling_rate = sampling_rate
        encoder_layers = self.mm.get_layer_ids("tokenizer_encoder")
        decoder_layers = self.mm.get_layer_ids("tokenizer_decoder")
        self._has_sharded = bool(encoder_layers or decoder_layers)
        self._standard_tokenizer = None

    def use_standard_tokenizer(self, tokenizer):
        self._standard_tokenizer = tokenizer

    @torch.inference_mode()
    def encode(self, audio_waveform):
        if self._standard_tokenizer is not None:
            result = self._standard_tokenizer.encode(audio_waveform.unsqueeze(0) if audio_waveform.dim() == 2 else audio_waveform)
            return result.audio_codes.squeeze(0)
        if self._has_sharded:
            return self._encode_layerwise(audio_waveform)
        raise RuntimeError("No audio tokenizer available. Load the standard tokenizer or run the model sharder first.")

    @torch.inference_mode()
    def decode(self, audio_tokens):
        if self._standard_tokenizer is not None:
            result = self._standard_tokenizer.decode(audio_tokens.unsqueeze(0))
            return result.audio_values[0]
        if self._has_sharded:
            return self._decode_layerwise(audio_tokens)
        raise RuntimeError("No audio tokenizer available. Load the standard tokenizer or run the model sharder first.")

    def _encode_layerwise(self, audio_waveform):
        encoder_ids = sorted(self.mm.get_layer_ids("tokenizer_encoder"))
        hidden_states = audio_waveform.float()
        for i, layer_id in enumerate(encoder_ids):
            if i + 1 < len(encoder_ids):
                self.mm.prefetch_layer(encoder_ids[i + 1])
            weights = self.mm.load_layer(layer_id)
            for key, w in weights.items():
                if "weight" in key and w.dim() == 2:
                    hidden_states = F.linear(hidden_states, w.to(hidden_states.dtype))
                    break
            self.mm.evict_layer(layer_id)
        return hidden_states.long()

    def _decode_layerwise(self, audio_tokens):
        decoder_ids = sorted(self.mm.get_layer_ids("tokenizer_decoder"))
        hidden_states = audio_tokens.float()
        for i, layer_id in enumerate(decoder_ids):
            if i + 1 < len(decoder_ids):
                self.mm.prefetch_layer(decoder_ids[i + 1])
            weights = self.mm.load_layer(layer_id)
            for key, w in weights.items():
                if "weight" in key and w.dim() == 2:
                    hidden_states = F.linear(hidden_states, w.to(hidden_states.dtype))
                    break
            self.mm.evict_layer(layer_id)
        return hidden_states
