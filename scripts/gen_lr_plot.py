import json
from pathlib import Path
import sys
import torch
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so project imports work when running this script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Project imports
from training.schedulers import build_scheduler


def main():
    exp_dir = Path('experiments/IN718/debug_x4_res')
    cfg_path = exp_dir / 'logs' / 'run_config.json'
    if not cfg_path.exists():
        cfg_path = exp_dir / 'config.json'
    cfg = json.loads(cfg_path.read_text())

    epochs = int(cfg.get('epochs', 100))
    base_lr = float(cfg.get('lr', 1e-3))

    # Create a dummy parameter so optimizer can be constructed
    param = torch.nn.Parameter(torch.zeros(1))
    optim_cfg = cfg.get('optimizer', {})
    opt_type = optim_cfg.get('type', 'AdamW')
    weight_decay = optim_cfg.get('weight_decay', 0.0)

    if opt_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW([param], lr=base_lr, weight_decay=weight_decay)
    elif opt_type.lower() == 'adam':
        optimizer = torch.optim.Adam([param], lr=base_lr, weight_decay=weight_decay)
    elif opt_type.lower() == 'sgd':
        optimizer = torch.optim.SGD([param], lr=base_lr, weight_decay=weight_decay)
    else:
        # Default
        optimizer = torch.optim.AdamW([param], lr=base_lr, weight_decay=weight_decay)

    # Build scheduler using project helper
    class CfgNamespace:
        def __init__(self, d):
            self.__dict__.update(d)
        def __getattr__(self, k):
            return self.__dict__.get(k)
    cfg_ns = CfgNamespace(cfg)

    scheduler = build_scheduler(optimizer, cfg_ns)

    lrs = []
    # Step scheduler for each epoch and record lr after stepping to match training behavior
    for epoch in range(epochs):
        if scheduler is not None:
            # PyTorch schedulers expect step() called each epoch
            scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs + 1), lrs, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    out = Path('learning_rate_debug_x4_res.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f'Wrote {out.resolve()}')


if __name__ == '__main__':
    main()
