#!/usr/bin/env python3
"""Evaluate completed few-shot 4x4 adaptation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "analysis/out/fewshot_4x4_manifest.json"
OUT_DIR = ROOT / "analysis/out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import metric_panel_hardened as mph  # noqa: E402


def _completed_sr_dir(run: dict) -> Path | None:
    exp_dir = ROOT / str(run["experiment"])
    summary = exp_dir / "inference/test_best/summary.json"
    sr_dir = exp_dir / "inference/test_best/sr_quaternions"
    if summary.exists() and sr_dir.exists():
        return sr_dir
    return None


def _target_groups(runs: list[dict]) -> OrderedDict[str, list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for run in runs:
        grouped.setdefault(str(run["target_key"]), []).append(run)
    return grouped


def _choose_ref_dir(runs: list[dict]) -> Path | None:
    for run in runs:
        if str(run["method"]) == "OCRP":
            sr_dir = _completed_sr_dir(run)
            if sr_dir is not None:
                return sr_dir
    for run in runs:
        sr_dir = _completed_sr_dir(run)
        if sr_dir is not None:
            return sr_dir
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-prefix", default="fewshot_4x4_hardened_metrics")
    parser.add_argument("--target", default=None, help="Optional target_key filter.")
    parser.add_argument(
        "--include-classical",
        action="store_true",
        help="Also evaluate nearest/bicubic/SLERP/Symm-SLERP on the same held-out split.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    runs = list(manifest["runs"])
    if args.target:
        runs = [run for run in runs if str(run["target_key"]) == args.target]

    rows: list[dict] = []
    skipped: list[dict] = []
    for target_key, group in _target_groups(runs).items():
        completed = [run for run in group if _completed_sr_dir(run) is not None]
        if not completed:
            skipped.append({"target_key": target_key, "reason": "no completed summaries"})
            continue

        ref_dir = _choose_ref_dir(group)
        if ref_dir is None:
            skipped.append({"target_key": target_key, "reason": "no reference LR/HR directory"})
            continue

        first = group[0]
        symmetry = str(first.get("symmetry", ""))
        if not symmetry:
            cfg_path = ROOT / str(first["experiment"]) / str(first["config"])
            cfg = json.loads(cfg_path.read_text())
            symmetry = str(cfg.get("symmetry_group", cfg.get("symmetry", "Oh")))
        label = (
            f"{first['target_name']} few-shot "
            f"({first['adaptation_samples']} train / {first['heldout_samples']} test)"
        )

        sym_quats = mph.configure_symmetry(symmetry)
        ops = mph.conjugated_ops(sym_quats)
        ids = mph.sample_ids(ref_dir)
        records = mph.preload_dataset_records(ref_dir, ids, ops)

        methods: OrderedDict[str, Path | None] = OrderedDict()
        if args.include_classical:
            methods.update((name, None) for name in mph.CLASSICAL_METHODS)
        for run in group:
            sr_dir = _completed_sr_dir(run)
            if sr_dir is None:
                skipped.append(
                    {
                        "target_key": target_key,
                        "method": run["method"],
                        "reason": "missing completed inference summary",
                    }
                )
                continue
            methods[str(run["method"])] = sr_dir

        print(f"{target_key}: {label}, {symmetry}, n={len(ids)}", flush=True)
        for method, method_dir in methods.items():
            print(f"  {method}", flush=True)
            rows.append(
                mph.summarize_method(
                    target_key,
                    label,
                    symmetry,
                    method,
                    method_dir,
                    records,
                    ops,
                )
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.out_prefix}.csv"
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    winners = mph.best_by_dataset(rows) if rows else []
    mph.write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "source_manifest": str(args.manifest),
                    "fewshot_fraction": manifest.get("fraction"),
                    "seed": manifest.get("seed"),
                    "epochs": manifest.get("epochs"),
                    "lr_scale": manifest.get("lr_scale"),
                    "partial_safe": True,
                    "completed_methods": len(rows),
                    "include_classical": bool(args.include_classical),
                },
                "rows": rows,
                "winners": winners,
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n"
    )
    mph.write_markdown(md_path, rows, winners)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete target/method entries.")


if __name__ == "__main__":
    main()
