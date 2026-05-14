import gc
import json
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

from clonemodel.lite.accent_engine import AccentEngine, AccentFingerprint
from clonemodel.lite.layerwise_engine import LayerWiseAudioTokenizer, LayerWiseForwardPass
from clonemodel.lite.memory_manager import TieredMemoryManager
from clonemodel.lite.quantize import load_quantized_layer

logger = logging.getLogger(__name__)

QUALITY_PRESETS = {
    "realtime": {"num_step": 1, "guidance_scale": 0.0},
    "fast": {"num_step": 2, "guidance_scale": 1.2},
    "balanced": {"num_step": 8, "guidance_scale": 1.8},
    "best": {"num_step": 16, "guidance_scale": 2.0},
}


class OmniVoiceLite:
    def __init__(self, shard_dir, config, max_memory_gb=3.5, device="cpu", enable_prefetch=True):
        self.shard_dir = Path(shard_dir)
        self.config = config
        self.device = device
        self.max_memory_gb = max_memory_gb

        self.memory_manager = TieredMemoryManager(
            shard_dir=shard_dir, max_memory_gb=max_memory_gb,
            enable_prefetch=enable_prefetch, device=device,
        )

        llm_config = config.get("llm_config", config)
        self.hidden_size = llm_config.get("hidden_size", 1024)
        self.num_layers = llm_config.get("num_hidden_layers", 28)
        self.num_audio_codebook = config.get("num_audio_codebook", 8)
        self.audio_vocab_size = config.get("audio_vocab_size", 1025)
        self.audio_mask_id = config.get("audio_mask_id", 1024)
        self.sampling_rate = 24000

        self._llm_engine = LayerWiseForwardPass(self.memory_manager, llm_config)
        self._audio_tokenizer = LayerWiseAudioTokenizer(self.memory_manager, config, self.sampling_rate)
        self._accent_engine = AccentEngine(self.sampling_rate)

        self._text_tokenizer = None
        self._load_text_tokenizer()

        self._text_embed_weights = None
        self._audio_embed_weights = None
        self._audio_heads_weights = None
        self._norm_weights = None
        self._codebook_offsets = torch.arange(self.num_audio_codebook) * self.audio_vocab_size
        self._duration_estimator = None
        self.progress_callback: Optional[Callable] = None

    @classmethod
    def from_pretrained(cls, shard_dir, max_memory_gb=3.5, device=None, enable_prefetch=True):
        shard_path = Path(os.path.expanduser(shard_dir))
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard directory not found: {shard_path}\nRun: python -m clonemodel.lite.shard_model --output {shard_path}")

        config_path = shard_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading OmniVoice Lite from {shard_path} on {device} (memory: {max_memory_gb:.1f}GB)")

        model = cls(shard_dir=str(shard_path), config=config, max_memory_gb=max_memory_gb, device=device, enable_prefetch=enable_prefetch)

        manifest_path = shard_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            logger.info(f"Model: {manifest.get('model_name', 'unknown')} | Layers: {len(manifest.get('layers', {}))} | INT{manifest.get('bits', '?')} | {manifest.get('total_quantized_mb', '?')}MB")

        return model

    def _load_text_tokenizer(self):
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            tok_path = self.shard_dir
            if (tok_path / "tokenizer.json").exists():
                try:
                    self._text_tokenizer = AutoTokenizer.from_pretrained(
                        str(tok_path),
                        local_files_only=True,
                    )
                except Exception as exc:
                    logger.warning(f"AutoTokenizer load failed for {tok_path}: {exc}")
                    self._text_tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=str(tok_path / "tokenizer.json"),
                        eos_token="<|im_end|>",
                        pad_token="<|endoftext|>",
                    )
        except Exception as exc:
            logger.warning(f"Text tokenizer initialization failed: {exc}")

    def _ensure_embeddings_loaded(self):
        if self._text_embed_weights is None:
            try:
                self._text_embed_weights = load_quantized_layer(self.shard_dir / "text_embeddings.pt", device=self.device)
            except FileNotFoundError:
                pass
        if self._audio_embed_weights is None:
            try:
                self._audio_embed_weights = load_quantized_layer(self.shard_dir / "audio_embeddings.pt", device=self.device)
            except FileNotFoundError:
                pass
        if self._audio_heads_weights is None:
            try:
                self._audio_heads_weights = load_quantized_layer(self.shard_dir / "audio_heads.pt", device=self.device)
            except FileNotFoundError:
                pass
        if self._norm_weights is None:
            try:
                self._norm_weights = load_quantized_layer(self.shard_dir / "llm_norm.pt", device=self.device)
            except FileNotFoundError:
                pass

    @torch.inference_mode()
    def generate(self, text, ref_audio=None, ref_text=None, language=None, instruct=None, dialect=None, quality="balanced", speed=None, duration=None, progress_callback=None):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        self.progress_callback = progress_callback
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
        num_step = preset["num_step"]
        guidance_scale = preset["guidance_scale"]

        total_start = time.monotonic()
        self._report_progress("init", 0, 1, "Starting generation...")

        accent_params = {}
        if ref_audio is not None and quality != "realtime":
            ref_wav = self._load_ref_audio(ref_audio)
            fingerprint = self._accent_engine.analyze_accent(ref_wav, hint_language=language, hint_dialect=dialect)
            accent_params = self._accent_engine.get_generation_params(fingerprint, target_language=language)

        if speed is None and "speed" in accent_params:
            speed = accent_params["speed"]
        if instruct is None and "instruct" in accent_params:
            instruct = accent_params["instruct"]
        if language is None and "language" in accent_params:
            language = accent_params["language"]

        self._ensure_embeddings_loaded()

        ref_audio_tokens = None
        if ref_audio is not None:
            self._report_progress("encode", 0, 1, "Encoding reference audio...")
            ref_wav = self._load_ref_audio(ref_audio)
            ref_audio_tokens = self._encode_reference(ref_wav)

        self._report_progress("prepare", 0, 1, "Preparing inputs...")
        input_data = self._prepare_inputs(text=text, ref_text=ref_text, ref_audio_tokens=ref_audio_tokens, language=language, instruct=instruct, speed=speed, duration=duration)

        self._report_progress("generate", 0, num_step, "Starting generation...")
        generated_tokens = self._iterative_generate(input_data=input_data, num_step=num_step, guidance_scale=guidance_scale)

        self._report_progress("decode", 0, 1, "Decoding audio...")
        audio = self._decode_tokens(generated_tokens)

        elapsed = time.monotonic() - total_start
        logger.info(f"Total generation time: {elapsed:.1f}s")

        self.memory_manager.clear_kv_cache()
        gc.collect()
        return audio

    def _load_ref_audio(self, ref_audio):
        if isinstance(ref_audio, str):
            import soundfile as sf
            audio, sr = sf.read(ref_audio, dtype='float32')  # Load as float32
            if sr != self.sampling_rate:
                import torchaudio
                audio_t = torch.from_numpy(audio).float()
                if audio_t.dim() == 1:
                    audio_t = audio_t.unsqueeze(0)
                audio_t = torchaudio.functional.resample(audio_t, sr, self.sampling_rate)
                audio = audio_t.numpy()
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
            return audio
        else:
            if ref_audio.ndim == 1:
                return ref_audio[np.newaxis, :]
            return ref_audio

    def _encode_reference(self, ref_wav):
        wav_tensor = torch.from_numpy(ref_wav).to(self.device)
        if wav_tensor.dim() == 2:
            wav_tensor = wav_tensor.unsqueeze(0)
        tokens = self._audio_tokenizer.encode(wav_tensor)
        del wav_tensor
        gc.collect()
        return tokens

    def _prepare_inputs(self, text, ref_text, ref_audio_tokens, language, instruct, speed, duration):
        if self._text_tokenizer is None:
            raise RuntimeError("Text tokenizer not loaded")

        if duration is not None:
            target_len = max(1, int(duration * 75))
        elif ref_text and ref_audio_tokens is not None:
            ref_chars = max(len(ref_text), 1)
            ref_frames = ref_audio_tokens.shape[-1]
            target_len = max(1, int(len(text) / ref_chars * ref_frames))
        else:
            target_len = max(1, int(len(text) * 3))

        if speed and speed > 0 and speed != 1.0:
            target_len = max(1, int(target_len / speed))

        # The Lite app runs on CPU, so keep cloned replies responsive.
        target_len = min(target_len, 48)

        style_text = ""
        if ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = language if language else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        style_ids = self._text_tokenizer(style_text, return_tensors="pt").input_ids.to(self.device)

        full_text = (ref_text.strip() + " " + text.strip()) if ref_text else text.strip()
        full_text = re.sub(r"[\r\n]+", "", full_text)
        full_text = re.sub(r"[ \t]+", " ", full_text)
        wrapped = f"<|text_start|>{full_text}<|text_end|>"
        text_ids = self._text_tokenizer(wrapped, return_tensors="pt").input_ids.to(self.device)

        return {"style_ids": style_ids, "text_ids": text_ids, "ref_audio_tokens": ref_audio_tokens, "target_len": target_len}

    def _iterative_generate(self, input_data, num_step, guidance_scale):
        target_len = input_data["target_len"]
        C = self.num_audio_codebook

        tokens = torch.full((1, C, target_len), self.audio_mask_id, dtype=torch.long, device=self.device)
        timesteps = self._get_timesteps(num_step)
        total_mask = target_len * C

        for step in range(num_step):
            self._report_progress("generate", step, num_step, f"Step {step+1}/{num_step}")

            cond_ids, cond_audio_mask = self._build_cond_input(input_data, tokens)
            hidden_cond = self._run_llm_forward(cond_ids, cond_audio_mask)

            # --- FIX: Build unconditional input with style, text, and audio tokens (no ref audio) ---
            style_ids = input_data["style_ids"]
            text_ids = input_data["text_ids"]
            style_cb = style_ids.repeat(1, C).view(1, C, -1)
            text_cb = text_ids.repeat(1, C).view(1, C, -1)
            # Unconditional input: [style, text, current_tokens]
            uncond_parts = [style_cb, text_cb, tokens]
            uncond_ids = torch.cat(uncond_parts, dim=2)
            uncond_seq_len = uncond_ids.shape[2]
            # Only the audio portion is marked True in the mask
            uncond_audio_mask = torch.zeros(1, uncond_seq_len, dtype=torch.bool, device=self.device)
            uncond_audio_mask[0, -target_len:] = True

            logger.debug(f"Step {step}: cond_ids {cond_ids.shape}, uncond_ids {uncond_ids.shape}, cond_audio_mask {cond_audio_mask.shape}, uncond_audio_mask {uncond_audio_mask.shape}")

            hidden_uncond = self._run_llm_forward(uncond_ids, uncond_audio_mask)

            cond_logits = self._apply_audio_heads(hidden_cond, target_len)
            uncond_logits = self._apply_audio_heads(hidden_uncond, target_len)

            if guidance_scale != 0:
                c_lp = F.log_softmax(cond_logits, dim=-1)
                u_lp = F.log_softmax(uncond_logits, dim=-1)
                log_probs = torch.log_softmax(c_lp + guidance_scale * (c_lp - u_lp), dim=-1)
            else:
                log_probs = F.log_softmax(cond_logits, dim=-1)

            log_probs[..., self.audio_mask_id] = -float("inf")
            pred_tokens = log_probs.argmax(dim=-1)
            confidence = log_probs.max(dim=-1).values

            layer_ids = torch.arange(C, device=self.device).view(1, -1, 1)
            confidence = confidence - (layer_ids * 5.0)

            if step == num_step - 1:
                k = int((tokens == self.audio_mask_id).sum().item())
            else:
                frac = timesteps[step + 1] - timesteps[step]
                k = min(math.ceil(total_mask * frac), int((tokens == self.audio_mask_id).sum().item()))

            if k <= 0:
                continue

            still_masked = (tokens == self.audio_mask_id)
            confidence.masked_fill_(~still_masked, -float("inf"))

            _, topk_idx = torch.topk(confidence.flatten(), k)
            flat_tokens = tokens.flatten()
            flat_pred = pred_tokens.flatten()
            flat_tokens[topk_idx] = flat_pred[topk_idx]
            tokens = flat_tokens.view(1, C, target_len)

            del hidden_cond, hidden_uncond, cond_logits, uncond_logits, log_probs, pred_tokens, confidence
            gc.collect()

        return tokens.squeeze(0)

    def _build_cond_input(self, input_data, current_tokens):
        style_ids = input_data["style_ids"]
        text_ids = input_data["text_ids"]
        C = self.num_audio_codebook

        style_cb = style_ids.repeat(1, C).view(1, C, -1)
        text_cb = text_ids.repeat(1, C).view(1, C, -1)

        parts = [style_cb, text_cb]
        if input_data["ref_audio_tokens"] is not None:
            ref = input_data["ref_audio_tokens"].unsqueeze(0).to(self.device)
            parts.append(ref)
        parts.append(current_tokens)

        cond_ids = torch.cat(parts, dim=2)
        total_len = cond_ids.shape[2]

        audio_start = total_len - current_tokens.shape[2]
        if input_data["ref_audio_tokens"] is not None:
            audio_start -= input_data["ref_audio_tokens"].shape[-1]

        audio_mask = torch.zeros(1, total_len, dtype=torch.bool, device=self.device)
        audio_mask[0, audio_start:] = True

        return cond_ids, audio_mask

    def _run_llm_forward(self, input_ids, audio_mask):
        inputs_embeds = self._prepare_embeddings(input_ids, audio_mask)
        hidden = self._llm_engine.run_forward(inputs_embeds=inputs_embeds, num_layers=self.num_layers, prefix="llm_layer")
        if self._norm_weights:
            hidden = self._llm_engine.apply_final_norm(hidden, self._norm_weights)
        return hidden

    def _prepare_embeddings(self, input_ids, audio_mask):
        if self._text_embed_weights is None or self._audio_embed_weights is None:
            raise RuntimeError("Embedding weights not loaded")

        text_embed_w = self._text_embed_weights.get("weight")
        if text_embed_w is None:
            for k, v in self._text_embed_weights.items():
                if v.dim() == 2:
                    text_embed_w = v
                    break
        if text_embed_w is None:
            raise RuntimeError("Cannot find text embedding weight")

        # Validate text embedding weight shape
        logger.debug(f"text_embed_w shape: {text_embed_w.shape}, expected (vocab_size, embed_dim)")
        if text_embed_w.shape[0] < text_embed_w.shape[1]:
            logger.warning(f"text_embed_w might be transposed! Shape {text_embed_w.shape} looks like (embed_dim, vocab_size)")
            text_embed_w = text_embed_w.t()

        text_ids = input_ids[:, 0, :] if input_ids.dim() == 3 else input_ids
        text_embed_w = text_embed_w.to(self.device).to(torch.float32)
        text_embeds = F.embedding(text_ids, text_embed_w)
        logger.debug(f"text_embeds shape: {text_embeds.shape}, expected (batch, seq_len, embed_dim)")

        audio_embed_w = None
        for k, v in self._audio_embed_weights.items():
            if v.dim() == 2:
                audio_embed_w = v.to(self.device).to(torch.float32)
                break

        if audio_embed_w is not None and input_ids.dim() == 3:
            logger.debug(f"audio_embed_w shape: {audio_embed_w.shape}, expected (vocab_size, embed_dim)")
            if audio_embed_w.shape[0] < audio_embed_w.shape[1]:
                logger.warning(f"audio_embed_w might be transposed! Shape {audio_embed_w.shape} looks like (embed_dim, vocab_size)")
                audio_embed_w = audio_embed_w.t()

            offsets = self._codebook_offsets.to(self.device).view(1, -1, 1)
            shifted = (input_ids * audio_mask.unsqueeze(1)) + offsets
            shifted = shifted.clamp(0, audio_embed_w.shape[0] - 1)

            # Log shapes for debugging
            try:
                audio_embed_raw = F.embedding(shifted, audio_embed_w)
                logger.debug(f"audio_embed_raw shape: {audio_embed_raw.shape}, expected (batch, C, seq_len, embed_dim)")
                audio_embeds = audio_embed_raw.sum(dim=1)
                logger.debug(f"audio_embeds after sum(dim=1): {audio_embeds.shape}, expected (batch, seq_len, embed_dim)")
                logger.debug(f"text_embeds shape: {text_embeds.shape}")
                logger.debug(f"audio_mask shape: {audio_mask.shape}, audio_mask.unsqueeze(-1): {audio_mask.unsqueeze(-1).shape}")

                # Validate shapes match before torch.where
                if audio_embeds.shape != text_embeds.shape:
                    logger.error(f"Shape mismatch: audio_embeds {audio_embeds.shape} != text_embeds {text_embeds.shape}")
                    # Try to fix by reshaping
                    if audio_embeds.shape[:-1] == text_embeds.shape[:-1]:
                        logger.info(f"Attempting to align embedding dimensions...")
                        min_dim = min(audio_embeds.shape[-1], text_embeds.shape[-1])
                        audio_embeds = audio_embeds[..., :min_dim]
                        text_embeds = text_embeds[..., :min_dim]
                        logger.info(f"Shapes after fix: audio_embeds {audio_embeds.shape}, text_embeds {text_embeds.shape}")
                    else:
                        raise RuntimeError(f"Audio and text embeddings have mismatched shapes: {audio_embeds.shape} vs {text_embeds.shape}")

            except Exception as e:
                logger.error(f"Shape error during audio embedding: {e}")
                logger.error(f"input_ids: {input_ids.shape}, audio_mask: {audio_mask.shape}")
                logger.error(f"shifted: {shifted.shape}, audio_embed_w: {audio_embed_w.shape}")
                raise

            embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)
        else:
            embeds = text_embeds

        logger.debug(f"Final embeds shape: {embeds.shape}, expected (batch, seq_len, embed_dim)")
        return embeds

    def _apply_audio_heads(self, hidden_states, target_len):
        if self._audio_heads_weights is None:
            raise RuntimeError("Audio heads weights not loaded")

        head_w = None
        for k, v in self._audio_heads_weights.items():
            if v.dim() == 2:
                head_w = v.to(self.device).to(hidden_states.dtype)
                break
        if head_w is None:
            raise RuntimeError("Cannot find audio heads weight")

        logger.debug(f"_apply_audio_heads: hidden_states shape {hidden_states.shape}, target_len {target_len}")
        logger.debug(f"_apply_audio_heads: head_w shape {head_w.shape}")

        target_hidden = hidden_states[:, -target_len:, :]
        logger.debug(f"_apply_audio_heads: target_hidden shape {target_hidden.shape}, expected (batch, {target_len}, hidden_dim)")

        try:
            logits_flat = F.linear(target_hidden, head_w)
        except RuntimeError as e:
            logger.error(f"F.linear error: {e}")
            logger.error(f"target_hidden shape: {target_hidden.shape}")
            logger.error(f"head_w shape: {head_w.shape}")
            # Try to provide more context and fallback reshape
            if target_hidden.dim() > 3:
                logger.error(f"target_hidden is {target_hidden.dim()}D, should be 3D (batch, seq, hidden)")
                target_hidden = target_hidden.reshape(target_hidden.size(0), -1, target_hidden.size(-1))
                logger.info(f"Reshaped target_hidden to {target_hidden.shape}")
                logits_flat = F.linear(target_hidden, head_w)
            elif target_hidden.shape[-1] != head_w.shape[1]:
                logger.error(f"Hidden dim {target_hidden.shape[-1]} does not match head_w {head_w.shape[1]}")
                min_dim = min(target_hidden.shape[-1], head_w.shape[1])
                target_hidden = target_hidden[..., :min_dim]
                head_w = head_w[:, :min_dim]
                logger.info(f"Reshaped for fallback: target_hidden {target_hidden.shape}, head_w {head_w.shape}")
                logits_flat = F.linear(target_hidden, head_w)
            else:
                raise RuntimeError(f"_apply_audio_heads failed: {e}\ntarget_hidden: {target_hidden.shape}, head_w: {head_w.shape}")

        B, T, _ = logits_flat.shape
        logits = logits_flat.view(B, T, self.num_audio_codebook, self.audio_vocab_size)
        logits = logits.permute(0, 2, 1, 3)
        return logits.float()

    def _decode_tokens(self, tokens):
        audio_tensor = self._audio_tokenizer.decode(tokens)
        if isinstance(audio_tensor, torch.Tensor):
            audio = audio_tensor.cpu().numpy()
        else:
            audio = np.array(audio_tensor)
        if audio.ndim > 1:
            audio = audio.squeeze()
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.5
        return audio

    def _get_timesteps(self, num_step, t_shift=0.1):
        ts = torch.linspace(0, 1, num_step + 1)
        ts = t_shift * ts / (1 + (t_shift - 1) * ts)
        return ts.tolist()

    def _report_progress(self, phase, step, total, message):
        if self.progress_callback:
            try:
                self.progress_callback(phase, step, total, message)
            except Exception:
                pass

    def set_audio_tokenizer(self, tokenizer):
        self._audio_tokenizer.use_standard_tokenizer(tokenizer)

    @property
    def supported_languages(self):
        try:
            from clonemodel.utils.lang_map import LANG_NAMES, lang_display_name
            return sorted(lang_display_name(n) for n in LANG_NAMES)
        except ImportError:
            return []

    @property
    def supported_dialects(self):
        return AccentEngine.get_supported_dialects()

    def cleanup(self):
        self.memory_manager.cleanup()
        self._text_embed_weights = None
        self._audio_embed_weights = None
        self._audio_heads_weights = None
        self._norm_weights = None
        gc.collect()
