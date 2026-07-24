#!/usr/bin/env python3
"""Render Table 3 as a zero-shot-only transfer table from metric CSVs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper/202607_Umang_EBSD_SR_paper__NMI/EBSD_SR_Nature_v4"
OUT_TEX = PAPER_DIR / "tables/table3_zero_shot.tex"

METHODS = [
    "Nearest",
    "Bicubic",
    "SLERP",
    "Symm-SLERP",
    "Atindama inpainting",
    "Q-RBSA-adapted",
    "QEDSR",
    "EDSR",
    "RCAN",
    "SAN",
    "HAN",
    "OCRP (ours)",
]

METHOD_LABELS = {
    "Atindama inpainting": r"Atindama inpainting~\cite{atindama2023restoration}",
    "Q-RBSA-adapted": r"\qrbsaadapted~\cite{jangid2024qrbsa}",
    "QEDSR": r"QEDSR~\cite{lim2017edsr}",
    "EDSR": r"EDSR~\cite{lim2017edsr}",
    "RCAN": r"RCAN~\cite{zhang2018rcan}",
    "SAN": r"SAN~\cite{dai2019san}",
    "HAN": r"HAN~\cite{niu2020han}",
    "OCRP (ours)": r"\textbf{OCRP (ours)}",
}

PARAMS_K = {
    "Nearest": "--",
    "Bicubic": "--",
    "SLERP": "--",
    "Symm-SLERP": "--",
    "Atindama inpainting": r"$25{,}784$",
    "Q-RBSA-adapted": r"$6{,}175$",
    "QEDSR": r"$1{,}815$",
    "EDSR": r"$7{,}241$",
    "RCAN": r"$15{,}594$",
    "SAN": r"$15{,}862$",
    "HAN": r"$16{,}073$",
    "OCRP (ours)": r"$\boldsymbol{49}$",
}

BLOCKS = [
    {
        "label": r"IN718 $\to$ CoNi (FCC): zero-shot test split $n=20$",
        "zero_csv": ROOT / "analysis/out/zeroshot_coni_4x4_all_baselines_metrics.csv",
    },
    {
        "label": r"Ti-6Al-4V $\to$ Ti7-deformed (HCP): zero-shot test split $n=20$",
        "zero_csv": ROOT / "analysis/out/zeroshot_ti7_deformed_4x4_all_baselines_metrics.csv",
    },
    # Ti64 is retained as a provenance/stress-test audit but omitted from the
    # headline zero-shot table until its adaptation behavior is resolved.
    # {
    #     "label": r"Ti-6Al-4V $\to$ Ti64 (HCP): zero-shot test split $n=72$",
    #     "zero_csv": ROOT / "analysis/out/zeroshot_ti64_dic_mclean_4x4_all_baselines_metrics.csv",
    # },
]


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    key: str
    digits: int
    higher_is_better: bool = False


PANEL_A = [
    Metric("mean", r"Mean ($^\circ$)", "mis_mean_deg", 2),
    Metric("median", r"Median ($^\circ$)", "mis_median_deg", 2),
    Metric("boundary_f1", r"Boundary F1", "boundary_f1", 3, True),
    Metric("interior", r"Interior ($^\circ$)", "interior_mean_deg", 2),
    Metric("boundary_band", r"Boundary band ($^\circ$)", "boundary_band_mean_deg", 2),
]

PANEL_B = [
    Metric("p90", r"p90 ($^\circ$)", "mis_p90_deg", 2),
    Metric("p95", r"p95 ($^\circ$)", "mis_p95_deg", 2),
    Metric("p99", r"p99 ($^\circ$)", "mis_p99_deg", 2),
    Metric("psnr", r"PSNR (dB)", "psnr_ipf_xyz_db", 2, True),
    Metric("ssim", r"SSIM", "ssim_ipf_xyz", 3, True),
]


def read_csv_by_method(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def value(row: dict[str, str] | None, key: str) -> float | None:
    if row is None or key not in row or row[key] == "":
        return None
    try:
        val = float(row[key])
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def winners(zero: dict[str, dict[str, str]], metrics: list[Metric]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        vals = [
            val
            for method in METHODS
            if (val := value(zero.get(method), metric.key)) is not None
        ]
        if vals:
            out[metric.name] = max(vals) if metric.higher_is_better else min(vals)
    return out


def fmt_number(val: float | None, digits: int, *, bold: bool) -> str:
    if val is None:
        return "--"
    text = f"{val:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def is_winner(val: float | None, best: float | None) -> bool:
    return val is not None and best is not None and abs(val - best) <= 1.0e-10


def metric_cell(row: dict[str, str] | None, metric: Metric, win: dict[str, float]) -> str:
    val = value(row, metric.key)
    return fmt_number(
        val,
        metric.digits,
        bold=is_winner(val, win.get(metric.name)),
    )


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def render_panel(*, panel_label: str, metrics: list[Metric], include_caption: bool) -> list[str]:
    ncols = 2 + len(metrics)
    lines: list[str] = []
    if include_caption:
        lines.extend(
            [
                r"\begin{table*}[p]",
                r"\caption{%",
                r"\textbf{Zero-shot transfer to out-of-distribution FCC and HCP targets.}",
                r"All entries use source-domain checkpoints without target retraining. Lower is better for angular metrics; higher is better for boundary F1, PSNR and SSIM. \textbf{Bold} marks the best available method in each column.}",
                r"\label{tab:zero_shot}",
                r"\centering",
                rf"\textbf{{{panel_label}}}",
            ]
        )
    else:
        lines.extend(
            [
                r"\begin{table*}[p]",
                r"\centering",
                rf"\textbf{{Table~\ref{{tab:zero_shot}}b. {panel_label}}}",
            ]
        )
    lines.extend(
        [
            r"\vspace{0.3em}",
            r"{\scriptsize",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{ll{'c' * len(metrics)}}}",
            r"\toprule",
            " & ".join(["Method", "Params (K)", *[metric.label for metric in metrics]]) + r" \\",
        ]
    )
    for block in BLOCKS:
        zero = read_csv_by_method(block["zero_csv"])
        win = winners(zero, metrics)
        lines.extend(
            [
                r"\midrule",
                rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{block['label']}}}}} \\",
                r"\midrule",
            ]
        )
        for idx, method in enumerate(METHODS):
            if idx == 4:
                lines.append(r"\midrule")
            row = zero.get(method)
            rendered = [
                method_label(method),
                PARAMS_K[method],
                *[metric_cell(row, metric, win) for metric in metrics],
            ]
            lines.append(" & ".join(rendered) + r" \\")
    lines.extend(
        [
            r"% Ti64 stress-audit block intentionally omitted from the headline zero-shot table for now.",
            r"\botrule",
            r"\end{tabular}",
            r"}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def render() -> str:
    lines: list[str] = []
    lines.extend(
        render_panel(
            panel_label="a. Orientation-accuracy and boundary metrics.",
            metrics=PANEL_A,
            include_caption=True,
        )
    )
    lines.extend(
        render_panel(
            panel_label="Error-tail percentiles and IPF image fidelity.",
            metrics=PANEL_B,
            include_caption=False,
        )
    )
    return "\n".join(lines)


def main() -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
