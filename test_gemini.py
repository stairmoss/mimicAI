#!/usr/bin/env python3
"""Test script for Gemini API integration."""
from gemini_helper import is_gemini_available, get_gemini_config, validate_gemini_request

def test_config():
    cfg = get_gemini_config()
    print("Gemini Config Test:", cfg)
    valid, err = validate_gemini_request()
    print("Gemini Valid:", valid, "Error:", err)

if __name__ == "__main__":
    test_config()
