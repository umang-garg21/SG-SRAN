#!/usr/bin/env python3
"""Launch prepared few-shot 4x4 train+inference jobs in tmux workers."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "analysis/out/fewshot_4x4_manifest.json"


def _replace_gpu(cmd: list[str], gpu: str) -> list[str]:
    out = list(cmd)
    for flag in ("--gpu", "--gpu_ids"):
        if flag in out:
            idx = out.index(flag)
            if idx + 1 >= len(out):
                raise ValueError(f"{flag} is missing a value in command: {cmd}")
            out[idx + 1] = str(gpu)
    return out


def _shell_command(cmd: list[str], gpu: str) -> str:
    env = f"MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu))}"
    return f"{env} " + shlex.join(cmd)


def _worker_script(
    *,
    worker_id: int,
    gpu: str,
    runs: list[dict],
    out_dir: Path,
    skip_existing: bool,
) -> Path:
    script_path = out_dir / f"worker_{worker_id:02d}_gpu{gpu}.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "",
        f"echo 'worker {worker_id} gpu {gpu} starting at '$(date)",
    ]

    for run in runs:
        exp_dir = ROOT / str(run["experiment"])
        logs_dir = exp_dir / "logs"
        train_log = logs_dir / "fewshot_tmux_train.log"
        infer_log = logs_dir / "fewshot_tmux_infer.log"
        summary = exp_dir / "inference/test_best/summary.json"
        best_ckpt = exp_dir / "checkpoints/best_model.pt"
        train_cmd = _replace_gpu(list(run["train_command"]), gpu)
        infer_cmd = _replace_gpu(list(run["inference_command"]), gpu)
        label = f"{run['target_name']} :: {run['method']}"

        lines.extend(
            [
                "",
                f"echo '--- {label} ---'",
                f"mkdir -p {shlex.quote(str(logs_dir))}",
            ]
        )
        if skip_existing:
            lines.extend(
                [
                    (
                        f"if [[ -f {shlex.quote(str(summary))} ]]; then "
                        f"echo 'skip completed {label}'; continue; fi"
                    ),
                    (
                        f"if [[ -f {shlex.quote(str(best_ckpt))} ]]; then "
                        f"echo 'checkpoint exists, running inference for {label}'; "
                        f"{_shell_command(infer_cmd, gpu)} > {shlex.quote(str(infer_log))} 2>&1; "
                        "continue; fi"
                    ),
                ]
            )
        lines.extend(
            [
                f"echo 'train start {label} '$(date)",
                f"{_shell_command(train_cmd, gpu)} > {shlex.quote(str(train_log))} 2>&1",
                f"touch {shlex.quote(str(logs_dir / 'fewshot_train.done'))}",
                f"echo 'infer start {label} '$(date)",
                f"{_shell_command(infer_cmd, gpu)} > {shlex.quote(str(infer_log))} 2>&1",
                f"touch {shlex.quote(str(logs_dir / 'fewshot_infer.done'))}",
                f"echo 'done {label} '$(date)",
            ]
        )

    lines.append(f"echo 'worker {worker_id} gpu {gpu} finished at '$(date)")
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--session-prefix", default="qsr_fewshot4x4")
    parser.add_argument("--target", default=None, help="Optional target_key filter.")
    parser.add_argument("--method", default=None, help="Optional method-name filter.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    runs = list(manifest["runs"])
    if args.target:
        runs = [run for run in runs if str(run["target_key"]) == args.target]
    if args.method:
        runs = [run for run in runs if str(run["method"]) == args.method]
    if not runs:
        raise SystemExit("No runs selected.")

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("No GPUs selected.")

    shards = [[] for _ in gpus]
    for idx, run in enumerate(runs):
        shards[idx % len(gpus)].append(run)

    out_dir = ROOT / "analysis/out/fewshot_4x4_tmux_workers"
    out_dir.mkdir(parents=True, exist_ok=True)

    launched = []
    for worker_id, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            continue
        script = _worker_script(
            worker_id=worker_id,
            gpu=gpu,
            runs=shard,
            out_dir=out_dir,
            skip_existing=bool(args.skip_existing),
        )
        session = f"{args.session_prefix}_{worker_id:02d}"
        launched.append((session, gpu, script, len(shard)))
        if not args.dry_run:
            subprocess.run(["tmux", "new-session", "-d", "-s", session, str(script)], check=True)

    for session, gpu, script, count in launched:
        action = "prepared" if args.dry_run else "launched"
        print(f"{action} {session}: gpu={gpu} jobs={count} script={script.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
