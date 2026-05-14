import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from clonemodel.lite.quantize import load_quantized_layer

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    available_mb: float = 0.0
    gpu_mb: float = 0.0
    ceiling_mb: float = 0.0


class TieredMemoryManager:
    def __init__(self, shard_dir: str, max_memory_gb: float = 3.5, enable_prefetch: bool = True, device: str = "cpu"):
        self.shard_dir = Path(shard_dir)
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.enable_prefetch = enable_prefetch
        self.device = device

        self._active_layer: Optional[Dict[str, torch.Tensor]] = None
        self._active_layer_id: Optional[str] = None

        self._prefetch_buffer: Optional[Dict[str, torch.Tensor]] = None
        self._prefetch_layer_id: Optional[str] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_ready = threading.Event()

        self._kv_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._load_times: List[float] = []
        self._total_loads = 0

    def get_memory_stats(self) -> MemoryStats:
        stats = MemoryStats(ceiling_mb=self.max_memory_bytes / (1024 * 1024))
        if _PSUTIL_AVAILABLE:
            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            stats.rss_mb = mem_info.rss / (1024 * 1024)
            stats.vms_mb = mem_info.vms / (1024 * 1024)
            stats.available_mb = psutil.virtual_memory().available / (1024 * 1024)
        if torch.cuda.is_available() and "cuda" in self.device:
            stats.gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        return stats

    def check_memory_pressure(self) -> bool:
        if not _PSUTIL_AVAILABLE:
            return False
        stats = self.get_memory_stats()
        used_ratio = stats.rss_mb / stats.ceiling_mb if stats.ceiling_mb > 0 else 0
        if used_ratio > 0.90:
            logger.warning(f"Memory pressure HIGH: {stats.rss_mb:.0f}MB / {stats.ceiling_mb:.0f}MB ({used_ratio:.0%})")
            return True
        return False

    def load_layer(self, layer_id: str) -> Dict[str, torch.Tensor]:
        with self._prefetch_lock:
            if self._prefetch_layer_id == layer_id and self._prefetch_buffer is not None:
                self._prefetch_ready.wait(timeout=30)
                self._active_layer = self._prefetch_buffer
                self._active_layer_id = layer_id
                self._prefetch_buffer = None
                self._prefetch_layer_id = None
                return self._active_layer

        start = time.monotonic()
        layer_path = self.shard_dir / f"{layer_id}.pt"
        if not layer_path.exists():
            raise FileNotFoundError(f"Shard file not found: {layer_path}. Run `python -m clonemodel.lite.shard_model` first.")

        self._active_layer = load_quantized_layer(layer_path, device=self.device)
        self._active_layer_id = layer_id
        elapsed = time.monotonic() - start
        self._load_times.append(elapsed)
        self._total_loads += 1
        return self._active_layer

    def evict_layer(self, layer_id: Optional[str] = None):
        if layer_id is not None and self._active_layer_id != layer_id:
            return
        if self._active_layer is not None:
            for key in list(self._active_layer.keys()):
                del self._active_layer[key]
            self._active_layer = None
            self._active_layer_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prefetch_layer(self, layer_id: str):
        if not self.enable_prefetch:
            return
        with self._prefetch_lock:
            if self._prefetch_layer_id == layer_id:
                return
        self._prefetch_ready.clear()

        def _load_in_background():
            try:
                layer_path = self.shard_dir / f"{layer_id}.pt"
                if not layer_path.exists():
                    return
                data = load_quantized_layer(layer_path, device=self.device)
                with self._prefetch_lock:
                    self._prefetch_buffer = data
                    self._prefetch_layer_id = layer_id
                self._prefetch_ready.set()
            except Exception as e:
                logger.error(f"Prefetch failed for {layer_id}: {e}")
                self._prefetch_ready.set()

        self._prefetch_thread = threading.Thread(target=_load_in_background, daemon=True)
        self._prefetch_thread.start()

    def store_kv_cache(self, layer_idx: int, kv: Dict[str, torch.Tensor]):
        self._kv_cache[layer_idx] = {
            k: v.cpu() if v.device.type != "cpu" else v for k, v in kv.items()
        }

    def fetch_kv_cache(self, layer_idx: int, device: Optional[str] = None) -> Optional[Dict[str, torch.Tensor]]:
        if layer_idx not in self._kv_cache:
            return None
        target = device or self.device
        return {k: v.to(target) for k, v in self._kv_cache[layer_idx].items()}

    def clear_kv_cache(self):
        for layer_idx in list(self._kv_cache.keys()):
            for key in list(self._kv_cache[layer_idx].keys()):
                del self._kv_cache[layer_idx][key]
            del self._kv_cache[layer_idx]
        self._kv_cache.clear()
        gc.collect()

    def get_layer_ids(self, prefix: str) -> List[str]:
        files = sorted(self.shard_dir.glob(f"{prefix}*.pt"))
        return [f.stem for f in files]

    def get_stats_summary(self) -> str:
        if not self._load_times:
            return "No layers loaded yet."
        avg = sum(self._load_times) / len(self._load_times)
        total = sum(self._load_times)
        mem = self.get_memory_stats()
        return f"Layers loaded: {self._total_loads} | Avg: {avg:.3f}s | Total I/O: {total:.1f}s | RAM: {mem.rss_mb:.0f}MB/{mem.ceiling_mb:.0f}MB"

    def cleanup(self):
        self.evict_layer()
        self.clear_kv_cache()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=5)
        with self._prefetch_lock:
            self._prefetch_buffer = None
            self._prefetch_layer_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
