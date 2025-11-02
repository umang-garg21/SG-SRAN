#!/usr/bin/env python3
"""
Profile the Reynolds_res_QSRNet model to identify computational bottlenecks.
"""
import torch
import time
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import json
from pathlib import Path

# Load model and config
from models import build_model
from training.config_utils import load_and_prepare_config
from utils.config_utils import ConfigNamespace


@contextmanager
def timer(name, timings):
    """Context manager to time code blocks."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    timings[name].append(elapsed)


class ProfilingWrapper(torch.nn.Module):
    """Wrap a module to profile its forward pass."""
    def __init__(self, module, name, timings):
        super().__init__()
        self.module = module
        self.name = name
        self.timings = timings
    
    def forward(self, *args, **kwargs):
        with timer(self.name, self.timings):
            return self.module(*args, **kwargs)


def instrument_model(model, timings):
    """Add profiling wrappers to key modules."""
    # Wrap EquivariantReynoldsWrap modules
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'EquivariantReynoldsWrap':
            # We'll manually instrument the forward method
            original_forward = module.forward
            
            def make_profiled_forward(mod, mod_name):
                def profiled_forward(x):
                    # Time the group lifting
                    B, C, *spatial = x.shape
                    G, Cg, _ = mod.group_tensor.shape
                    n_feats = C // Cg
                    x_view = x.view(B, n_feats, Cg, *spatial)
                    
                    with timer(f"{mod_name}.lift_einsum", timings):
                        gamma_x = torch.einsum("gci,bni...->bgnc...", mod.group_tensor, x_view).reshape(
                            B * G, n_feats * Cg, *spatial
                        )
                    
                    # Time the wrapped function
                    with timer(f"{mod_name}.fn", timings):
                        fx = mod.fn(gamma_x)
                    
                    # Time the projection back
                    BG, Cout, *spatial_out = fx.shape
                    n_out = Cout // Cg
                    fx_view = fx.view(B, G, n_out, Cg, *spatial_out)
                    
                    with timer(f"{mod_name}.project_einsum", timings):
                        fx_proj = torch.einsum("gci,bgni...->bgnc...", mod.group_tensor_inv, fx_view)
                    
                    # Time the group averaging
                    with timer(f"{mod_name}.group_mean", timings):
                        result = fx_proj.mean(dim=1).reshape(B, Cout, *spatial_out)
                    
                    return result
                
                return profiled_forward
            
            module.forward = make_profiled_forward(module, name)


def profile_model(config_path, num_iterations=10, batch_size=4, device='cuda'):
    """Profile the model and report timing statistics."""
    print(f"Loading config from: {config_path}")
    cfg = load_and_prepare_config(Path(config_path))
    
    # Override batch size
    cfg.batch_size = batch_size
    
    print(f"Building model: {cfg.model.type}")
    model = build_model(cfg)
    model = model.to(device)
    model.eval()
    
    # Create timings dict
    timings = defaultdict(list)
    
    # Instrument the model
    print("Instrumenting model with profiling hooks...")
    instrument_model(model, timings)
    
    # Create dummy input (quaternion, so 4 channels)
    # Assuming scale=4, input would be H/4 x W/4
    input_size = 64  # patch size after downsampling
    dummy_input = torch.randn(batch_size, 4, input_size, input_size, device=device)
    
    print(f"\nWarming up with 3 iterations...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    print(f"\nProfiling with {num_iterations} iterations...")
    with torch.no_grad():
        for i in range(num_iterations):
            with timer("total_forward", timings):
                _ = model(dummy_input)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations")
    
    # Compute statistics
    print("\n" + "="*80)
    print("PROFILING RESULTS (milliseconds)")
    print("="*80)
    print(f"{'Operation':<50} {'Mean':>10} {'Std':>10} {'% Total':>10}")
    print("-"*80)
    
    total_time = np.mean(timings['total_forward']) * 1000  # to ms
    
    # Sort by mean time (descending)
    sorted_ops = sorted(timings.items(), key=lambda x: np.mean(x[1]), reverse=True)
    
    for name, times in sorted_ops:
        times_ms = np.array(times) * 1000  # to ms
        mean_time = np.mean(times_ms)
        std_time = np.std(times_ms)
        pct = (mean_time / total_time) * 100 if total_time > 0 else 0
        print(f"{name:<50} {mean_time:>10.3f} {std_time:>10.3f} {pct:>9.1f}%")
    
    print("-"*80)
    print(f"{'TOTAL':<50} {total_time:>10.3f}")
    print("="*80)
    
    # Group analysis
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    einsum_time = 0
    fn_time = 0
    mean_time = 0
    
    for name, times in timings.items():
        mean = np.mean(times) * 1000
        if 'einsum' in name:
            einsum_time += mean
        elif '.fn' in name and 'einsum' not in name:
            fn_time += mean
        elif 'mean' in name:
            mean_time += mean
    
    print(f"Time in group tensor operations (einsum):  {einsum_time:>10.3f} ms ({einsum_time/total_time*100:>5.1f}%)")
    print(f"Time in wrapped functions (convs, etc):    {fn_time:>10.3f} ms ({fn_time/total_time*100:>5.1f}%)")
    print(f"Time in group averaging:                   {mean_time:>10.3f} ms ({mean_time/total_time*100:>5.1f}%)")
    print("="*80)
    
    # Memory usage
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("MEMORY USAGE")
        print("="*80)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print("="*80)
    
    return timings


if __name__ == "__main__":
    import sys
    
    config_path = "experiments/IN718/debug_x4_eqv_res/config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    timings = profile_model(
        config_path=config_path,
        num_iterations=20,
        batch_size=4,
        device=device
    )
    
    print("\n✓ Profiling complete!")
