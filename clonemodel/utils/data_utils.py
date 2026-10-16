#!/usr/bin/env python3

"""Data utilities for batch inference and evaluation.

Provides ``read_test_list()`` to parse JSONL test list files used by
``MimicAI.cli.infer_batch`` and evaluation scripts.
"""

import json
import logging
from pathlib import Path


def read_test_list(path):
    """Read a JSONL test list file.

    Each line should be a JSON object.  Only ``id`` and ``text`` are required;
    all other fields are optional (default to ``None``):
        id, text, ref_audio, ref_text, instruct,
        language_id, language_name, duration, speed

    Note: ``language_name`` is only used by evaluation scripts (under
    ``MimicAI/eval/``) for grouping and reporting results.  The model
    itself only consumes ``language_id``.

    Returns a list of dicts.
    """
    path = Path(path)
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON at line {line_no}: {line}")
                continue

            sample = {
                "id": obj.get("id"),
                "text": obj.get("text"),
                "ref_audio": obj.get("ref_audio"),
                "ref_text": obj.get("ref_text"),
                "language_id": obj.get("language_id"),
                "language_name": obj.get("language_name"),
                "duration": obj.get("duration"),
                "speed": obj.get("speed"),
                "instruct": obj.get("instruct"),
            }
            samples.append(sample)
    return samples
