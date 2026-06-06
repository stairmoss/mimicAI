#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add current dir to python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    from voice_manager import _is_safe_id, VoiceManager
except ImportError as e:
    print(f"[-] Failed to import voice_manager: {e}")
    sys.exit(1)

def test_safe_id():
    print("[*] Running _is_safe_id validation tests...")
    
    # Valid IDs
    assert _is_safe_id("valid_id-123") == True, "Failed: valid_id-123 should be allowed"
    assert _is_safe_id("another123SafeID") == True, "Failed: another123SafeID should be allowed"
    assert _is_safe_id("a") == True, "Failed: single char should be allowed"
    
    # Invalid / Unsafe IDs
    assert _is_safe_id("") == False, "Failed: empty string should be rejected"
    assert _is_safe_id(None) == False, "Failed: None should be rejected"
    assert _is_safe_id("../etc/passwd") == False, "Failed: path traversal should be rejected"
    assert _is_safe_id("dir/subdir") == False, "Failed: slashes should be rejected"
    assert _is_safe_id("profile; rm -rf") == False, "Failed: commands should be rejected"
    assert _is_safe_id("abc.wav") == False, "Failed: dots should be rejected"
    assert _is_safe_id("voice id") == False, "Failed: spaces should be rejected"
    
    print("[+] _is_safe_id validation tests passed!")

def test_path_traversal_prevention():
    print("[*] Running path traversal prevention tests...")
    vm = VoiceManager()
    
    # Setup temporary voices_dir
    original_voices_dir = vm.voices_dir
    temp_dir = Path(__file__).parent / "temp_test_voices"
    temp_dir.mkdir(exist_ok=True)
    vm.voices_dir = temp_dir
    
    try:
        # Test delete_voice with traversal
        assert vm.delete_voice("../dummy") == False, "Failed: delete_voice should reject traversal ID"
        
        # Test get_reference_audio_path with traversal
        assert vm.get_reference_audio_path("../dummy") == None, "Failed: get_reference_audio_path should reject traversal ID"
        
        # Test get_voice_metadata with traversal
        assert vm.get_voice_metadata("../dummy") == None, "Failed: get_voice_metadata should reject traversal ID"
        
    finally:
        # Cleanup
        vm.voices_dir = original_voices_dir
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    print("[+] Path traversal prevention tests passed!")

def test_voice_design_validation():
    print("[*] Running voice design validation tests...")
    vm = VoiceManager()
    
    # Non-dict input
    assert vm._generate_designed_voice("test", None) is None, "Failed: None design input should return None"
    assert vm._generate_designed_voice("test", "not-a-dict") is None, "Failed: String design input should return None"
    
    # Verify malicious/invalid parameters do not crash the engine
    try:
        vm._generate_designed_voice("test", {
            "gender": "unknown-malicious",
            "accent": "malicious-accent",
            "speed": "not-a-float"
        })
    except Exception as exc:
        assert False, f"Failed: Invalid fields in design dict should not raise exception: {exc}"
        
    print("[+] Voice design validation tests passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("           MIMICAI SECURITY TEST SUITE")
    print("=" * 60)
    try:
        test_safe_id()
        test_path_traversal_prevention()
        test_voice_design_validation()
        print("\n[SUCCESS] All security validations are functioning correctly!")
    except AssertionError as e:
        print(f"\n[FAILURE] Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during tests: {e}")
        sys.exit(1)
