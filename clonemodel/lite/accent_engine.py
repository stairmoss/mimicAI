import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

KERALA_DIALECTS = {
    "thiruvananthapuram": {"description": "Southern Kerala", "speed_factor": 0.95, "pitch_shift": -0.5, "emphasis_pattern": "end", "intonation": "rising"},
    "kochi": {"description": "Central Kerala", "speed_factor": 1.05, "pitch_shift": 0.0, "emphasis_pattern": "even", "intonation": "neutral"},
    "kozhikode": {"description": "Northern Kerala (Calicut)", "speed_factor": 1.10, "pitch_shift": 0.5, "emphasis_pattern": "start", "intonation": "falling"},
    "thrissur": {"description": "Thrissur", "speed_factor": 1.0, "pitch_shift": 0.3, "emphasis_pattern": "even", "intonation": "rising"},
    "kannur": {"description": "Kannur", "speed_factor": 1.15, "pitch_shift": 0.8, "emphasis_pattern": "start", "intonation": "falling"},
    "kasaragod": {"description": "Kasaragod (Tulu/Kannada influenced)", "speed_factor": 1.0, "pitch_shift": 0.2, "emphasis_pattern": "even", "intonation": "neutral"},
    "palakkad": {"description": "Palakkad (Tamil influenced)", "speed_factor": 0.95, "pitch_shift": -0.3, "emphasis_pattern": "end", "intonation": "neutral"},
    "idukki": {"description": "Idukki/High Range", "speed_factor": 0.90, "pitch_shift": 0.0, "emphasis_pattern": "even", "intonation": "neutral"},
}

INDIAN_ACCENTS = {
    "malayalam": {"lang_id": "ml", "instruct_hint": "indian accent", "speed_range": (0.85, 1.15), "pitch_range": (-1.0, 1.0), "sub_dialects": KERALA_DIALECTS},
    "tamil": {"lang_id": "ta", "instruct_hint": "indian accent", "speed_range": (0.90, 1.10), "pitch_range": (-0.5, 0.5)},
    "telugu": {"lang_id": "te", "instruct_hint": "indian accent", "speed_range": (0.85, 1.10), "pitch_range": (-0.5, 1.0)},
    "kannada": {"lang_id": "kn", "instruct_hint": "indian accent", "speed_range": (0.90, 1.05), "pitch_range": (-0.5, 0.5)},
    "hindi": {"lang_id": "hi", "instruct_hint": "indian accent", "speed_range": (0.90, 1.15), "pitch_range": (-0.5, 1.0)},
    "bengali": {"lang_id": "bn", "instruct_hint": "indian accent", "speed_range": (0.85, 1.05), "pitch_range": (-1.0, 0.5)},
    "marathi": {"lang_id": "mr", "instruct_hint": "indian accent", "speed_range": (0.90, 1.10), "pitch_range": (-0.5, 0.5)},
    "gujarati": {"lang_id": "gu", "instruct_hint": "indian accent", "speed_range": (0.90, 1.10), "pitch_range": (0.0, 1.0)},
}


@dataclass
class AccentFingerprint:
    speaking_rate: float = 1.0
    syllable_rate: float = 0.0
    pause_frequency: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0
    pitch_contour: str = "neutral"
    energy_mean: float = 0.0
    energy_std: float = 0.0
    emphasis_pattern: str = "even"
    detected_language: Optional[str] = None
    detected_dialect: Optional[str] = None
    confidence: float = 0.0

    def to_speed_factor(self) -> float:
        if self.syllable_rate > 0:
            return max(0.5, min(2.0, self.syllable_rate / 4.0))
        return self.speaking_rate


class AccentEngine:
    def __init__(self, sampling_rate: int = 24000):
        self.sampling_rate = sampling_rate

    def analyze_accent(self, audio, hint_language=None, hint_dialect=None):
        if audio.ndim > 1:
            audio = audio.squeeze()
        audio = audio.astype(np.float32)

        fp = AccentFingerprint()
        if _LIBROSA_AVAILABLE:
            fp = self._analyze_prosody(audio, fp)
        else:
            fp = self._analyze_basic(audio, fp)

        if hint_language:
            fp.detected_language = hint_language.lower()
        if hint_dialect:
            fp.detected_dialect = hint_dialect.lower()

        fp = self._match_dialect_profile(fp)
        return fp

    def get_generation_params(self, fingerprint, target_language=None):
        params = {}
        speed = fingerprint.to_speed_factor()

        if fingerprint.detected_dialect:
            dialect_info = KERALA_DIALECTS.get(fingerprint.detected_dialect)
            if dialect_info:
                speed *= dialect_info["speed_factor"]

        params["speed"] = max(0.5, min(2.0, speed))

        instruct_parts = []
        lang_lower = (target_language or "").lower()
        if lang_lower in ("english", "en") or not target_language:
            lang_info = INDIAN_ACCENTS.get(fingerprint.detected_language or "", {})
            if lang_info and "instruct_hint" in lang_info:
                instruct_parts.append(lang_info["instruct_hint"])

        if fingerprint.pitch_mean > 0:
            if fingerprint.pitch_mean > 250:
                instruct_parts.append("high pitch")
            elif fingerprint.pitch_mean < 150:
                instruct_parts.append("low pitch")

        if instruct_parts:
            params["instruct"] = ", ".join(instruct_parts)

        if target_language:
            params["language"] = target_language
        elif fingerprint.detected_language:
            lang_info = INDIAN_ACCENTS.get(fingerprint.detected_language, {})
            if "lang_id" in lang_info:
                params["language"] = fingerprint.detected_language.title()

        return params

    def _analyze_prosody(self, audio, fp):
        sr = self.sampling_rate
        try:
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=600, sr=sr)
            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                fp.pitch_mean = float(np.mean(voiced_f0))
                fp.pitch_std = float(np.std(voiced_f0))
                fp.pitch_range = float(np.max(voiced_f0) - np.min(voiced_f0))
                if len(voiced_f0) > 10:
                    half = len(voiced_f0) // 2
                    diff = np.mean(voiced_f0[half:]) - np.mean(voiced_f0[:half])
                    if diff > fp.pitch_std * 0.3:
                        fp.pitch_contour = "rising"
                    elif diff < -fp.pitch_std * 0.3:
                        fp.pitch_contour = "falling"
        except Exception:
            pass

        try:
            rms = librosa.feature.rms(y=audio, frame_length=2048)[0]
            fp.energy_mean = float(np.mean(rms))
            fp.energy_std = float(np.std(rms))
            if len(rms) > 10:
                third = len(rms) // 3
                start_energy = np.mean(rms[:third])
                end_energy = np.mean(rms[-third:])
                if start_energy > end_energy * 1.3:
                    fp.emphasis_pattern = "start"
                elif end_energy > start_energy * 1.3:
                    fp.emphasis_pattern = "end"
        except Exception:
            pass

        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            duration_s = len(audio) / sr
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if duration_s > 0:
                fp.syllable_rate = len(onsets) / duration_s
                fp.speaking_rate = fp.syllable_rate / 3.5
            intervals = librosa.effects.split(audio, top_db=30)
            num_pauses = max(0, len(intervals) - 1)
            fp.pause_frequency = num_pauses / max(duration_s, 0.1)
        except Exception:
            pass

        return fp

    def _analyze_basic(self, audio, fp):
        rms = np.sqrt(np.mean(audio ** 2))
        fp.energy_mean = float(rms)
        fp.speaking_rate = 1.0
        return fp

    def _match_dialect_profile(self, fp):
        if not fp.detected_language:
            return fp
        lang_info = INDIAN_ACCENTS.get(fp.detected_language, {})
        sub_dialects = lang_info.get("sub_dialects", {})
        if not sub_dialects or fp.detected_dialect:
            return fp

        best_score = -1
        best_dialect = None

        for dialect_name, profile in sub_dialects.items():
            score = 0
            if fp.speaking_rate > 0:
                score += max(0, 1.0 - abs(fp.speaking_rate - profile["speed_factor"]))
            if fp.pitch_mean > 0:
                score += max(0, 1.0 - abs((fp.pitch_mean - 200) / 100 - profile["pitch_shift"]))
            if fp.emphasis_pattern == profile["emphasis_pattern"]:
                score += 1.0
            if fp.pitch_contour == profile["intonation"]:
                score += 1.0
            if score > best_score:
                best_score = score
                best_dialect = dialect_name

        if best_dialect and best_score > 1.5:
            fp.detected_dialect = best_dialect
            fp.confidence = min(1.0, best_score / 4.0)

        return fp

    @staticmethod
    def get_supported_dialects():
        result = {}
        for lang, info in INDIAN_ACCENTS.items():
            dialects = list(info.get("sub_dialects", {}).keys())
            if dialects:
                result[lang] = dialects
        return result

    @staticmethod
    def get_dialect_info(language, dialect):
        lang_info = INDIAN_ACCENTS.get(language.lower(), {})
        return lang_info.get("sub_dialects", {}).get(dialect.lower())
