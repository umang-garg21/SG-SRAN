#!/usr/bin/env python
"""
Test script to verify reproducible seeding
"""
import random
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.seed_utils import set_seed

def test_reproducibility():
    """Test that seeding produces reproducible results"""
    
    print("="*80)
    print("Testing Reproducibility with Seeding")
    print("="*80)
    
    # Test 1: Same seed should give same results
    print("\n[Test 1] Same seed (42) → Same random numbers")
    
    set_seed(42)
    python_rand_1 = random.random()
    numpy_rand_1 = np.random.rand()
    torch_rand_1 = torch.rand(1).item()
    
    set_seed(42)
    python_rand_2 = random.random()
    numpy_rand_2 = np.random.rand()
    torch_rand_2 = torch.rand(1).item()
    
    assert python_rand_1 == python_rand_2, "Python random not reproducible!"
    assert numpy_rand_1 == numpy_rand_2, "NumPy random not reproducible!"
    assert torch_rand_1 == torch_rand_2, "PyTorch random not reproducible!"
    
    print(f"  Python:  {python_rand_1:.10f} == {python_rand_2:.10f} ✓")
    print(f"  NumPy:   {numpy_rand_1:.10f} == {numpy_rand_2:.10f} ✓")
    print(f"  PyTorch: {torch_rand_1:.10f} == {torch_rand_2:.10f} ✓")
    
    # Test 2: Different seeds should give different results
    print("\n[Test 2] Different seeds → Different random numbers")
    
    set_seed(42)
    result_seed_42 = torch.rand(1).item()
    
    set_seed(123)
    result_seed_123 = torch.rand(1).item()
    
    assert result_seed_42 != result_seed_123, "Different seeds should give different results!"
    
    print(f"  Seed 42:  {result_seed_42:.10f}")
    print(f"  Seed 123: {result_seed_123:.10f} ✓")
    
    # Test 3: Model initialization is reproducible
    print("\n[Test 3] Model initialization reproducibility")
    
    set_seed(42)
    model1 = torch.nn.Linear(10, 5)
    weights1 = model1.weight.data.clone()
    
    set_seed(42)
    model2 = torch.nn.Linear(10, 5)
    weights2 = model2.weight.data.clone()
    
    assert torch.allclose(weights1, weights2), "Model weights not reproducible!"
    print(f"  Model 1 weight sum: {weights1.sum().item():.10f}")
    print(f"  Model 2 weight sum: {weights2.sum().item():.10f} ✓")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nYour training will be fully reproducible when using the same seed!")

if __name__ == "__main__":
    test_reproducibility()
