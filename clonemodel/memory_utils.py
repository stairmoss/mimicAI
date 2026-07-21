"""Memory and Garbage Collection Utilities."""
import gc

def free_unused_memory():
    gc.collect()
