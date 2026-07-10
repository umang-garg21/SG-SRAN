#!/usr/bin/env python3
"""Render Table 3 as zero-shot / seed-42 few-shot entries from metric CSVs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Paper/202607_Umang_EBSD_SR_paper__NMI/EBSD_SR_Nature_v4"
OUT_TEX = PAPER_DIR / "tables/table3_zero_fewshot.tex"

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
        "label": r"IN718 $\to$ CoNi (FCC): zero-shot test split $n=20$ / seed-42 few-shot split, $2$ adaptation and $18$ held-out test",
        "zero_csv": ROOT / "analysis/out/zeroshot_coni_4x4_all_baselines_metrics.csv",
        "few_dataset": "coni_x250",
    },
    {
        "label": r"Ti-6Al-4V $\to$ Ti7-deformed (HCP): zero-shot test split $n=20$ / seed-42 few-shot split, $2$ adaptation and $18$ held-out test",
        "zero_csv": ROOT / "analysis/out/zeroshot_ti7_deformed_4x4_all_baselines_metrics.csv",
        "few_dataset": "ti7_deformed",
    },
    # Ti64 is retained as a provenance/stress-test audit but omitted from the
    # headline zero-shot/adaptation table until its adaptation behavior is resolved.
    # {
    #     "label": r"Ti-6Al-4V $\to$ Ti64 (HCP): zero-shot test split $n=72$ / seed-42 few-shot split, $73$ adaptation and $653$ held-out test",
    #     "zero_csv": ROOT / "analysis/out/zeroshot_ti64_dic_mclean_4x4_all_baselines_metrics.csv",
    #     "few_dataset": "ti64_dic_mclean",
    # },
]


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    zero_key: str
    few_key: str
    digits: int
    higher_is_better: bool = False


PANEL_A = [
    Metric("mean", r"Mean ($^\circ$)", "mis_mean_deg", "mean_deg", 2),
    Metric("median", r"Median ($^\circ$)", "mis_median_deg", "median_deg", 2),
    Metric("boundary_f1", r"Boundary F1", "boundary_f1", "boundary_f1", 3, True),
    Metric("interior", r"Interior ($^\circ$)", "interior_mean_deg", "interior_mean_deg", 2),
    Metric("boundary_band", r"Boundary band ($^\circ$)", "boundary_band_mean_deg", "boundary_band_mean_deg", 2),
]

PANEL_B = [
    Metric("p90", r"p90 ($^\circ$)", "mis_p90_deg", "p90_deg", 2),
    Metric("p95", r"p95 ($^\circ$)", "mis_p95_deg", "p95_deg", 2),
    Metric("p99", r"p99 ($^\circ$)", "mis_p99_deg", "p99_deg", 2),
    Metric("psnr", r"PSNR (dB)", "psnr_ipf_xyz_db", "psnr_ipf_xyz_db", 2, True),
    Metric("ssim", r"SSIM", "ssim_ipf_xyz", "ssim_ipf_xyz", 3, True),
]


def read_csv_by_method(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def read_fewshot(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = "OCRP (ours)" if row["method"] == "OCRP" else row["method"]
            out[(row["dataset"], method)] = row
    return out


def value(row: dict[str, str] | None, key: str) -> float | None:
    if row is None or key not in row or row[key] == "":
        return None
    try:
        val = float(row[key])
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def winners(
    zero: dict[str, dict[str, str]],
    few: dict[tuple[str, str], dict[str, str]],
    few_dataset: str,
    metrics: list[Metric],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for metric in metrics:
        zero_vals = [
            val
            for method in METHODS
            if (val := value(zero.get(method), metric.zero_key)) is not None
        ]
        few_vals = [
            val
            for method in METHODS
            if (val := value(few.get((few_dataset, method)), metric.few_key)) is not None
        ]
        if zero_vals:
            out[(metric.name, "zero")] = (
                max(zero_vals) if metric.higher_is_better else min(zero_vals)
            )
        if few_vals:
            out[(metric.name, "few")] = (
                max(few_vals) if metric.higher_is_better else min(few_vals)
            )
    return out


def fmt_number(val: float | None, digits: int, *, bold: bool) -> str:
    if val is None:
        return "--"
    text = f"{val:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def is_winner(val: float | None, best: float | None) -> bool:
    return val is not None and best is not None and abs(val - best) <= 1.0e-10


def slash_cell(
    zero_row: dict[str, str] | None,
    few_row: dict[str, str] | None,
    metric: Metric,
    win: dict[tuple[str, str], float],
) -> str:
    zero_val = value(zero_row, metric.zero_key)
    few_val = value(few_row, metric.few_key)
    zero_text = fmt_number(
        zero_val,
        metric.digits,
        bold=is_winner(zero_val, win.get((metric.name, "zero"))),
    )
    few_text = fmt_number(
        few_val,
        metric.digits,
        bold=is_winner(few_val, win.get((metric.name, "few"))),
    )
    return f"{zero_text} / {few_text}"


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def render_panel(
    *,
    panel_label: str,
    metrics: list[Metric],
    few: dict[tuple[str, str], dict[str, str]],
    include_caption: bool,
) -> list[str]:
    ncols = 2 + len(metrics)
    lines: list[str] = []
    if include_caption:
        lines.extend(
            [
                r"\begin{table*}[p]",
                r"\caption{%",
                r"\textbf{Zero-shot transfer and seed-42 few-shot adaptation on out-of-distribution targets.}",
                r"Each metric cell is reported as zero-shot / few-shot. Zero-shot uses the source-domain checkpoint without target retraining; few-shot uses the complete seed-42 adaptation audit on the held-out target split after removing adaptation samples. Deterministic rows in the few-shot half are evaluated on the same held-out split without training. Lower is better for angular metrics; higher is better for boundary F1, PSNR and SSIM. \textbf{Bold} marks the best available method separately for the zero-shot and few-shot value in each cell.}",
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
        win = winners(zero, few, block["few_dataset"], metrics)
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
            zero_row = zero.get(method)
            few_row = few.get((block["few_dataset"], method))
            row = [
                method_label(method),
                PARAMS_K[method],
                *[slash_cell(zero_row, few_row, metric, win) for metric in metrics],
            ]
            lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"% Ti64 stress-audit block intentionally omitted from the headline zero-shot/adaptation table for now.",
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
    few = read_fewshot(ROOT / "analysis/out/fewshot_4x4_hardened_metrics_s42.csv")
    lines: list[str] = []
    lines.extend(
        render_panel(
            panel_label="a. Orientation-accuracy and boundary metrics.",
            metrics=PANEL_A,
            few=few,
            include_caption=True,
        )
    )
    lines.extend(
        render_panel(
            panel_label="Error-tail percentiles and IPF image fidelity.",
            metrics=PANEL_B,
            few=few,
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
