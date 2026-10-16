#!/usr/bin/env python3

"""
This module contains the text normalization function for WER evaluation. 
Copied from https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/workflows/dataprep/text_tools.py
"""

import re
import unicodedata

from unidecode import unidecode

import omnivoice.eval.wer.norm_config_module as norm_config_module

norm_config = norm_config_module.norm_config


def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code :
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)


    if config["lower_case"] and lower_case:
        text = text.lower()


    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)


    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)


    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += "]"

    normalized_text = re.sub(punct_pattern, " ", text)


    delete_patten = r"[" + config["del_set"] + "]"

    normalized_text = re.sub(delete_patten, "", normalized_text)


    if remove_numbers:

        digits_pattern = "[" + config["digit_set"]

        digits_pattern += "]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + "$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        normalized_text = unidecode(normalized_text)

    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text
