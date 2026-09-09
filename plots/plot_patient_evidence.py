#!/usr/bin/env python
"""Build a source-traceable longitudinal patient-level evidence-chain figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "figure_inputs" / "figure7_patient_evidence"
OUT = ROOT / "outputs" / "figure7_patient_evidence"

LABELS = ["heart_failure", "renal_failure", "infection", "pneumonia", "cerebrovascular", "diabetic_foot"]
DISPLAY = {
    "heart_failure": "Heart failure",
    "renal_failure": "Renal failure",
    "infection": "Infection",
    "pneumonia": "Pneumonia",
    "cerebrovascular": "Cerebrovascular",
    "diabetic_foot": "Diabetic foot",
}
COLORS = {
    "heart_failure": "#2F5D7C", "renal_failure": "#208C83",
    "infection": "#C65A46", "pneumonia": "#D49A35",
    "cerebrovascular": "#7D6AA2", "diabetic_foot": "#4F9AA8",
    "ink": "#20252B", "muted": "#77838D", "grid": "#E2E7E9",
    "risk": "#9E3D5B", "up": "#C65A46", "down": "#3C719D",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2, "axes.titlesize": 8.5, "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.7,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def clean(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.6, width=0.65, pad=2)
    ax.grid(axis=grid, color=COLORS["grid"], lw=0.65)


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.11, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")


def parse_top(value: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in json.loads(value).items()}


def load() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    row = pd.read_csv(DATA / "Figure7_case_source.csv").iloc[0]
    risk = pd.read_csv(DATA / "Figure7_risk_trajectory_source.csv").sort_values("prefix_frac")
    drivers = pd.read_csv(DATA / "Figure7_feature_driver_source.csv").copy()
    return row, risk, drivers


def timeline(ax: plt.Axes, row: pd.Series, risk: pd.DataFrame) -> None:
    x = risk["prefix_frac"].to_numpy(float) * 100
    truth = set(str(row["true_labels"]).split("|"))
    predicted = set(str(row["predicted_labels"]).split("|"))
    for label in LABELS:
        y = risk[f"risk_{label}"].to_numpy(float)
        important = label in truth or label in predicted
        ax.plot(x, y, color=COLORS[label], lw=2.0 if important else 1.0,
                ls="-" if important else "--", marker="o" if important else None,
                ms=3.6, alpha=0.98 if important else 0.52, zorder=3 if important else 2)
        if important:
            ax.text(101.8, y[-1], DISPLAY[label], color=COLORS[label], fontsize=6.5, va="center")
    for left, right, alpha in [(0, 25, .10), (25, 50, .16), (50, 75, .22), (75, 100, .28)]:
        ax.axvspan(left, right, color="#A9C8D0", alpha=alpha, lw=0, zorder=0)
    ax.axvline(25, color=COLORS["risk"], lw=1.1, ls=(0, (3, 2)), zorder=1)
    ax.annotate("earliest evaluated prefix", xy=(25, 0.996), xytext=(31, 1.055),
                fontsize=6.4, color=COLORS["risk"], ha="left",
                arrowprops={"arrowstyle": "-", "lw": .7, "color": COLORS["risk"]})
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 1.10)
    ax.set_xticks(x, ["25%", "50%", "75%", "100%"])
    ax.set_yticks([0, .25, .5, .75, 1.0], ["0", ".25", ".50", ".75", "1.00"])
    ax.set_ylabel("Predicted risk")
    ax.set_xlabel("Observed fraction of the admission record")
    ax.set_title("Longitudinal multi-label risk profile", loc="left", fontweight="bold", pad=6)
    clean(ax, "both")
    panel(ax, "a")


def context(ax: plt.Axes, row: pd.Series, drivers: pd.DataFrame) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.02, .88, "REPRESENTATIVE CASE", color=COLORS["muted"], fontsize=5.6, fontweight="bold")
    ax.text(.02, .76, f"Age {int(row['age'])} years", color=COLORS["ink"], fontsize=9.0, fontweight="bold")
    ax.plot([.02, .93], [.67, .67], color=COLORS["grid"], lw=.8)
    records = [("VITAL SIGNS", int(row["event_count_vitals"])), ("LABORATORY", int(row["event_count_labs"])),
               ("MEDICATION", int(row["event_count_meds"]))]
    for i, (name, value) in enumerate(records):
        y = .53 - i * .18
        ax.text(.02, y, name, color=COLORS["muted"], fontsize=5.6, fontweight="bold", va="center")
        ax.text(.93, y, f"{value:,}", color=COLORS["ink"], fontsize=8.0, ha="right", va="center")
        if i < 2:
            ax.plot([.02, .93], [y-.09, y-.09], color=COLORS["grid"], lw=.7)
    ax.text(.02, .04, "Structured record volume", fontsize=5.8, color=COLORS["muted"])
    panel(ax, "b")


def drivers_plot(ax: plt.Axes, drivers: pd.DataFrame) -> None:
    df = drivers.sort_values("z_score").copy()
    y = np.arange(len(df))
    color = [COLORS["up"] if bool(v) else COLORS["down"] for v in df["risk_direction"]]
    ax.barh(y, df["z_score"], height=.58, color=color, alpha=.9)
    ax.axvline(0, color=COLORS["muted"], lw=.75, ls=(0, (2, 2)))
    for yi, rec in zip(y, df.itertuples()):
        pos = rec.z_score
        label = f"{rec.value:.1f}"
        if pos > 5:
            ax.text(pos-.20, yi, label, ha="right", va="center", color="white", fontsize=6.0)
        else:
            ax.text(pos + (.22 if pos >= 0 else -.22), yi, label, ha="left" if pos >= 0 else "right", va="center", fontsize=6.0)
    ax.set_yticks(y, df["display"])
    ax.set_xlabel("Standardised value vs. reference")
    ax.set_title("Structured evidence at final evaluation", loc="left", fontweight="bold", pad=6)
    ax.set_xlim(-2.4, max(17.5, float(df["z_score"].max()) + 1.2))
    clean(ax)
    panel(ax, "c")


def outcomes(ax: plt.Axes, row: pd.Series, risk: pd.DataFrame) -> None:
    final = risk.iloc[-1]
    true = set(str(row["true_labels"]).split("|"))
    predicted = set(str(row["predicted_labels"]).split("|"))
    rows = []
    for label in LABELS:
        status = "TP" if label in true and label in predicted else "TN"
        rows.append((label, float(final[f"risk_{label}"]), status))
    rows.sort(key=lambda z: z[1])
    y = np.arange(len(rows))
    for yi, (label, value, status) in zip(y, rows):
        alpha = .96 if status == "TP" else .42
        ax.hlines(yi, 0, value, color=COLORS[label], lw=4.0, alpha=alpha)
        ax.plot(value, yi, "o", color=COLORS[label], ms=4.8, alpha=alpha)
        ax.text(min(value+.035, .945), yi, f"{value:.2f}", va="center", fontsize=6.2)
        ax.text(1.03, yi, status, va="center", ha="left", fontweight="bold", fontsize=6.2,
                color=COLORS["up"] if status == "TP" else COLORS["down"])
    ax.set_yticks(y, [DISPLAY[z[0]] for z in rows])
    ax.set_xlim(0, 1.15); ax.set_xlabel("Risk at final evaluation")
    ax.set_title("Outcome-level prediction", loc="left", fontweight="bold", pad=6)
    clean(ax)
    panel(ax, "d")


def draw_arrow(fig: plt.Figure) -> None:
    arrow = FancyArrowPatch((.16, .060), (.86, .060), transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=10, lw=1.25, color="#8798A1")
    fig.add_artist(arrow)
    fig.text(.16, .074, "progressive record accrual", fontsize=6.2, color=COLORS["muted"], ha="left")
    fig.text(.86, .074, "final multi-label decision", fontsize=6.2, color=COLORS["muted"], ha="right")


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Build the fixed patient-level evidence-chain figure (no options).")
        return
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    row, risk, drivers = load()
    risk.to_csv(OUT / "Figure7_risk_trajectory_source.csv", index=False)
    drivers.to_csv(OUT / "Figure7_feature_driver_source.csv", index=False)
    pd.DataFrame([row]).to_csv(OUT / "Figure7_case_source.csv", index=False)
    fig = plt.figure(figsize=(7.25, 5.55))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.25, 1.0], left=.10, right=.94, top=.84, bottom=.14, hspace=.74, wspace=.72)
    timeline(fig.add_subplot(gs[0, :]), row, risk)
    context(fig.add_subplot(gs[1, 0:2]), row, drivers)
    drivers_plot(fig.add_subplot(gs[1, 3:7]), drivers)
    outcomes(fig.add_subplot(gs[1, 8:]), row, risk)
    fig.suptitle("Patient-level evidence chain: progressive multi-organ deterioration", x=.10, ha="left", y=.97, fontsize=10.0, fontweight="bold")
    fig.text(.10, .925, "Representative high-risk true-positive case: heart failure, renal failure, infection and diabetic foot.", fontsize=6.7, color=COLORS["muted"])
    draw_arrow(fig)
    save(fig, "patient_level_longitudinal_evidence_chain")
    plt.close(fig)


if __name__ == "__main__":
    main()
