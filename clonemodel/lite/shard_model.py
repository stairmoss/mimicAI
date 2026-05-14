import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    logger.info(f"Downloading model: {name_or_path}...")
    return snapshot_download(name_or_path)


def _extract_llm_layers(model_path: str) -> dict:
    from safetensors import safe_open

    model_dir = Path(model_path)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        st_files = sorted(model_dir.glob("model*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    logger.info(f"Found {len(st_files)} safetensors files")

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    layers = {}
    embeddings = {}
    audio_embeddings = {}
    audio_heads = {}
    norm_params = {}
    other_params = {}

    for st_file in st_files:
        logger.info(f"Reading {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if "llm.layers." in key or "llm.model.layers." in key:
                    parts = key.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            layer_idx = int(parts[i + 1])
                            relative_key = ".".join(parts[i + 2:])
                            layer_key = f"llm_layer_{layer_idx:02d}"
                            if layer_key not in layers:
                                layers[layer_key] = {}
                            layers[layer_key][relative_key] = tensor
                            break

                elif "audio_embeddings" in key:
                    relative_key = key.split("audio_embeddings.")[-1] if "audio_embeddings." in key else "weight"
                    audio_embeddings[relative_key] = tensor

                elif "audio_heads" in key:
                    relative_key = key.split("audio_heads.")[-1] if "audio_heads." in key else "weight"
                    audio_heads[relative_key] = tensor

                elif any(x in key for x in ["embed_tokens", "llm.embed_tokens", "llm.model.embed_tokens"]):
                    relative_key = key.split(".")[-1]
                    embeddings[relative_key] = tensor

                elif any(x in key for x in ["llm.norm", "llm.model.norm", "model.norm"]):
                    relative_key = key.split(".")[-1]
                    norm_params[relative_key] = tensor

                else:
                    other_params[key] = tensor

                del tensor
                gc.collect()

    result = {
        "text_embeddings": embeddings,
        "audio_embeddings": audio_embeddings,
        "audio_heads": audio_heads,
        "llm_norm": norm_params,
        "config": config,
    }
    result.update(layers)

    if other_params:
        result["other"] = other_params

    logger.info(f"Extracted: {len(layers)} LLM layers, embeddings({len(embeddings)}), audio_embeddings({len(audio_embeddings)}), audio_heads({len(audio_heads)}), norm({len(norm_params)}), other({len(other_params)})")
    return result


def _extract_audio_tokenizer_layers(tokenizer_path: str) -> dict:
    from safetensors import safe_open

    model_dir = Path(tokenizer_path)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        return {}

    logger.info(f"Found {len(st_files)} audio tokenizer safetensors files")

    encoder_layers = {}
    decoder_layers = {}
    other_params = {}

    for st_file in st_files:
        logger.info(f"Reading tokenizer file: {st_file.name}...")
        with safe_open(str(st_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if "encoder" in key and "layers." in key:
                    parts = key.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                relative_key = ".".join(parts[i + 2:])
                                layer_key = f"tokenizer_encoder_{layer_idx:02d}"
                                if layer_key not in encoder_layers:
                                    encoder_layers[layer_key] = {}
                                encoder_layers[layer_key][relative_key] = tensor
                            except ValueError:
                                other_params[key] = tensor
                            break

                elif "decoder" in key and "layers." in key:
                    parts = key.split(".")
                    for i, part in enumerate(parts):
                        if part == "layers" and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                relative_key = ".".join(parts[i + 2:])
                                layer_key = f"tokenizer_decoder_{layer_idx:02d}"
                                if layer_key not in decoder_layers:
                                    decoder_layers[layer_key] = {}
                                decoder_layers[layer_key][relative_key] = tensor
                            except ValueError:
                                other_params[key] = tensor
                            break
                else:
                    other_params[key] = tensor

                del tensor
                gc.collect()

    result = {}
    result.update(encoder_layers)
    result.update(decoder_layers)
    if other_params:
        result["tokenizer_other"] = other_params

    logger.info(f"Audio tokenizer: {len(encoder_layers)} encoder layers, {len(decoder_layers)} decoder layers, {len(other_params)} other params")
    return result


def shard_model(model_name: str, output_dir: str, bits: int = 4, group_size: int = 128, shard_tokenizer: bool = True):
    from clonemodel.lite.quantize import save_quantized_layer, estimate_quantized_size

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    logger.info("=" * 60)
    logger.info("Phase 1: Sharding OmniVoice LLM")
    logger.info("=" * 60)

    model_path = _resolve_model_path(model_name)
    extracted = _extract_llm_layers(model_path)

    config = extracted.pop("config", {})
    config_out = output_path / "config.json"
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)

    manifest = {"model_name": model_name, "bits": bits, "group_size": group_size, "layers": {}}
    total_original = 0
    total_quantized = 0

    for layer_id, state_dict in extracted.items():
        if layer_id in ("config",) or not state_dict:
            continue

        layer_path = output_path / f"{layer_id}.pt"
        sizes = estimate_quantized_size(state_dict, bits=bits, group_size=group_size)
        total_original += sizes["original_mb"]
        total_quantized += sizes["quantized_mb"]

        logger.info(f"  {layer_id}: {sizes['original_mb']:.1f}MB -> ~{sizes['quantized_mb']:.1f}MB (INT{bits})")
        save_quantized_layer(state_dict, layer_path, bits=bits, group_size=group_size)

        manifest["layers"][layer_id] = {
            "file": f"{layer_id}.pt",
            "original_mb": round(sizes["original_mb"], 2),
            "quantized_mb": round(sizes["quantized_mb"], 2),
            "num_params": sum(t.numel() for t in state_dict.values()),
        }

        del state_dict
        gc.collect()

    del extracted
    gc.collect()

    if shard_tokenizer:
        logger.info("=" * 60)
        logger.info("Phase 2: Sharding Audio Tokenizer (HiggsAudioV2)")
        logger.info("=" * 60)

        tokenizer_path = os.path.join(model_path, "audio_tokenizer")
        if not os.path.isdir(tokenizer_path):
            try:
                tokenizer_path = _resolve_model_path("eustlb/higgs-audio-v2-tokenizer")
            except Exception:
                tokenizer_path = None

        if tokenizer_path and os.path.isdir(tokenizer_path):
            tokenizer_layers = _extract_audio_tokenizer_layers(tokenizer_path)

            for layer_id, state_dict in tokenizer_layers.items():
                if not state_dict:
                    continue
                layer_path = output_path / f"{layer_id}.pt"
                sizes = estimate_quantized_size(state_dict, bits=bits, group_size=group_size)
                total_original += sizes["original_mb"]
                total_quantized += sizes["quantized_mb"]

                logger.info(f"  {layer_id}: {sizes['original_mb']:.1f}MB -> ~{sizes['quantized_mb']:.1f}MB (INT{bits})")
                save_quantized_layer(state_dict, layer_path, bits=bits, group_size=group_size)

                manifest["layers"][layer_id] = {
                    "file": f"{layer_id}.pt",
                    "original_mb": round(sizes["original_mb"], 2),
                    "quantized_mb": round(sizes["quantized_mb"], 2),
                    "num_params": sum(t.numel() for t in state_dict.values()),
                }
                del state_dict
                gc.collect()

            del tokenizer_layers
            gc.collect()

            for config_file in ["config.json", "preprocessor_config.json"]:
                src = Path(tokenizer_path) / config_file
                if src.exists():
                    import shutil
                    shutil.copy2(src, output_path / f"tokenizer_{config_file}")

    manifest["total_original_mb"] = round(total_original, 2)
    manifest["total_quantized_mb"] = round(total_quantized, 2)
    manifest["compression_ratio"] = round(total_original / max(total_quantized, 1), 2)

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    text_tok_dir = Path(model_path)
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]:
        src = text_tok_dir / tok_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_path / tok_file)

    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info(f"Sharding complete! Output: {output_path}")
    logger.info(f"Layers: {len(manifest['layers'])} | Original: {total_original:.0f}MB | Quantized: {total_quantized:.0f}MB | {manifest['compression_ratio']:.1f}x | Time: {elapsed:.1f}s")
    logger.info("=" * 60)
    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Shard OmniVoice model into per-layer INT4 files")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--output", default=os.path.expanduser("~/.clonemodel_lite/shards"))
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-tokenizer", action="store_true")
    args = parser.parse_args()

    shard_model(model_name=args.model, output_dir=args.output, bits=args.bits, group_size=args.group_size, shard_tokenizer=not args.no_tokenizer)


if __name__ == "__main__":
    main()
