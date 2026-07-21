"""Memory and Garbage Collection Utilities."""
import gc

def free_unused_memory():
    gc.collect()


def check_system_memory() -> dict:
    return {"status": "normal", "gc": True}
