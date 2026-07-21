#!/usr/bin/env python3
"""Memory check verification script."""
from clonemodel.memory_utils import free_unused_memory, check_system_memory

def test_mem():
    free_unused_memory()
    print("Memory Check:", check_system_memory())

if __name__ == "__main__":
    test_mem()
