"""External-validation figure."""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace


# ==============================================================================
# Figure 4: external validation
# ==============================================================================

def _load_external():
    """Export Figure 5A and 5B with matched dimensions for side-by-side placement."""
    from io import BytesIO
    from pathlib import Path

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image


    ROOT = Path(__file__).resolve().parents[2]
    SOURCE = ROOT / "data" / "derived" / "figure4_transport" / "operating_points"
    OUT = ROOT / "outputs" / "figure4_transport_horizontal_pair"

    INK = "#222222"
    MUTED = "#606060"
    GRID = "#E9F0F1"
    BLUE = "#9EAAD1"
    AQUA = "#CDE2E8"
    PEACH = "#F5DBB6"
    LILAC = "#DACFE5"
    CORAL = "#F59790"

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.4,
        "axes.labelsize": 9.3,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.85,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


    def save(fig: plt.Figure, stem: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "pdf", "png", "tiff"):
            path = OUT / f"{stem}.{ext}"
            if ext == "svg":
                fig.savefig(path, bbox_inches="tight", facecolor="white")
            elif ext == "pdf":
                fig.savefig(path, bbox_inches="tight", facecolor="white")
            elif ext == "png":
                fig.savefig(path, dpi=450, bbox_inches="tight", facecolor="white")
            else:
                memory = BytesIO()
                fig.savefig(memory, format="png", dpi=600, bbox_inches="tight", facecolor="white")
                memory.seek(0)
                with Image.open(memory) as image:
                    image.convert("RGB").save(path, compression="tiff_lzw", dpi=(600, 600))
        plt.close(fig)


    def style(ax: plt.Axes) -> None:
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.75, color=MUTED)


    def transfer() -> None:
        data = pd.read_csv(SOURCE / "r4ab_unified_transfer_and_adaptation_source_data.csv")
        states = ["MIMIC-IV\ninternal", "eICU\nzero-shot", "eICU + 20%\nlocal update", "eICU + full\nlocal update"]
        x = np.arange(len(states))
        width = 0.29
        auroc = data.loc[data.metric.eq("AUROC"), "value"].to_numpy()
        auprc = data.loc[data.metric.eq("AUPRC"), "value"].to_numpy()
        fig, ax = plt.subplots(figsize=(7.25, 4.25))
        ax.axvspan(0.5, 1.5, color="#F3F0F5", zorder=0)
        bars_a = ax.bar(x - width / 2, auroc, width, color=BLUE, label="AUROC", zorder=3)
        bars_p = ax.bar(x + width / 2, auprc, width, color=PEACH, label="AUPRC", zorder=3)
        ci = pd.read_csv(SOURCE / "r4ab_internal_bootstrap_ci_source_data.csv").set_index("metric")
        for idx, metric_values in enumerate((auroc, auprc)):
            metric = ("AUROC", "AUPRC")[idx]
            row = ci.loc[metric]
            ax.errorbar(x[0] + (-width / 2 if idx == 0 else width / 2), metric_values[0], yerr=[[metric_values[0] - row.ci_low], [row.ci_high - metric_values[0]]], fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.8, zorder=5)
        for bars in (bars_a, bars_p):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.024, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        ax.annotate("Zero-shot transfer\nperformance drop", xy=(1, 0.60), xytext=(1, 0.97), ha="center", va="top", fontsize=7.8, color=CORAL, arrowprops={"arrowstyle": "-|>", "lw": 0.85, "color": CORAL})
        ax.annotate("20% of local data reaches\n99% of full-update AUROC", xy=(2, 0.702), xytext=(2.65, 0.94), ha="center", va="top", fontsize=7.5, color=BLUE, arrowprops={"arrowstyle": "-", "lw": 0.8, "color": BLUE})
        ax.set(xlim=(-0.55, 3.58), ylim=(0, 1.04), xticks=x, xticklabels=states, ylabel="Macro performance")
        ax.set_title("Cross-hospital transfer and adaptation", loc="left", fontsize=13.0, fontweight="bold", color=INK, pad=8)
        ax.legend(loc="upper left", ncol=2, fontsize=8.0, handlelength=1.2)
        style(ax)
        fig.text(0.125, 0.025, "Error bars: bootstrap 95% CI. Local updates used 5,550 (20%) or 27,749 (100%) target-domain sequences.", fontsize=6.8, color=MUTED)
        fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.22)
        save(fig, "Figure4A_cross_hospital_transfer_adaptation_horizontal")


    def transfer_narrow() -> None:
        """Compact version matched to the near-square operating-point panel."""
        data = pd.read_csv(SOURCE / "r4ab_unified_transfer_and_adaptation_source_data.csv")
        states = ["MIMIC-IV\ninternal", "eICU\nzero-shot", "eICU + 20%\nlocal update", "eICU + full\nlocal update"]
        x = np.arange(len(states))
        width = 0.27
        auroc = data.loc[data.metric.eq("AUROC"), "value"].to_numpy()
        auprc = data.loc[data.metric.eq("AUPRC"), "value"].to_numpy()
        fig, ax = plt.subplots(figsize=(5.25, 4.25))
        ax.axvspan(0.5, 1.5, color="#F3F0F5", zorder=0)
        bars_a = ax.bar(x - width / 2, auroc, width, color=BLUE, label="AUROC", zorder=3)
        bars_p = ax.bar(x + width / 2, auprc, width, color=PEACH, label="AUPRC", zorder=3)
        ci = pd.read_csv(SOURCE / "r4ab_internal_bootstrap_ci_source_data.csv").set_index("metric")
        for idx, metric_values in enumerate((auroc, auprc)):
            metric = ("AUROC", "AUPRC")[idx]
            row = ci.loc[metric]
            ax.errorbar(x[0] + (-width / 2 if idx == 0 else width / 2), metric_values[0], yerr=[[metric_values[0] - row.ci_low], [row.ci_high - metric_values[0]]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.3, zorder=5)
        for bars in (bars_a, bars_p):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.024, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
        ax.annotate("Zero-shot\nperformance drop", xy=(1, 0.60), xytext=(1, 0.95), ha="center", va="top", fontsize=6.7, color=CORAL, arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": CORAL})
        ax.annotate("20% local data reaches\n99% of full AUROC", xy=(2, 0.702), xytext=(2.62, 0.93), ha="center", va="top", fontsize=6.2, color=BLUE, arrowprops={"arrowstyle": "-", "lw": 0.7, "color": BLUE})
        ax.set(xlim=(-0.52, 3.52), ylim=(0, 1.04), xticks=x, xticklabels=states, ylabel="Macro performance")
        ax.set_title("Cross-hospital transfer and adaptation", loc="left", fontsize=10.8, fontweight="bold", color=INK, pad=7)
        ax.legend(loc="upper left", ncol=2, fontsize=7.0, handlelength=1.15)
        ax.tick_params(axis="x", labelsize=7.0)
        style(ax)
        fig.text(0.13, 0.025, "Error bars: bootstrap 95% CI. Local updates used 5,550 (20%) or 27,749 (100%) sequences.", fontsize=5.2, color=MUTED)
        fig.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.22)
        save(fig, "Figure4A_cross_hospital_transfer_adaptation_narrow")


    def operating_points() -> None:
        curve = pd.read_csv(SOURCE / "r4cd_global_threshold_sweep_source_data.csv").dropna(subset=["ppv"])
        points = (
            pd.read_csv(SOURCE / "r4cd_operating_point_profile_source_data.csv")
            .groupby("operating_point", as_index=False)
            .tail(1)
        )
        fig, ax = plt.subplots(figsize=(7.25, 4.25))
        x = curve.alert_rate.to_numpy() * 100
        ax.plot(x, curve.sensitivity.to_numpy() * 100, color=AQUA, linewidth=2.5, label="Sensitivity")
        ax.plot(x, curve.ppv.to_numpy() * 100, color=CORAL, linewidth=2.5, label="Positive predictive value")
        point_style = {"Fixed 0.5": ("Fixed threshold = 0.5", (42.5, 75.2), (43, 82), CORAL), "F1": ("Validation F1-tuned", (73.4, 83.5), (66, 96), AQUA), "F2": ("Validation F2-tuned", (92.0, 96.1), (91, 88), AQUA)}
        for row in points.itertuples(index=False):
            xx = row.alert_rate * 100
            sens = row.sensitivity * 100
            ppv = row.ppv * 100
            ax.scatter([xx], [sens], s=55, color=AQUA, zorder=4)
            ax.scatter([xx], [ppv], s=55, color=CORAL, zorder=4)
            label, _, _, color = point_style[row.operating_point]
            if row.operating_point == "Fixed 0.5":
                ax.annotate(f"{label}\nPPV {ppv:.1f}%", xy=(xx, ppv), xytext=(43, 83), ha="center", fontsize=7.9, bbox={"facecolor": "white", "edgecolor": CORAL, "pad": 0.35}, arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.9})
                ax.annotate(f"{label}\nSensitivity {sens:.1f}%", xy=(xx, sens), xytext=(43, 43), ha="center", fontsize=7.9, bbox={"facecolor": "white", "edgecolor": AQUA, "pad": 0.35}, arrowprops={"arrowstyle": "-", "color": AQUA, "lw": 0.9})
            elif row.operating_point == "F1":
                ax.annotate(f"{label}\nSensitivity {sens:.1f}%", xy=(xx, sens), xytext=(70, 96), ha="center", fontsize=7.9, bbox={"facecolor": "white", "edgecolor": AQUA, "pad": 0.35}, arrowprops={"arrowstyle": "-", "color": AQUA, "lw": 0.9})
                ax.annotate(f"{label}\nPPV {ppv:.1f}%", xy=(xx, ppv), xytext=(75, 54), ha="center", fontsize=7.9, bbox={"facecolor": "white", "edgecolor": CORAL, "pad": 0.35}, arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.9})
            else:
                ax.annotate(f"{label}\nSensitivity {sens:.1f}%", xy=(xx, sens), xytext=(92, 88), ha="center", fontsize=7.9, bbox={"facecolor": "white", "edgecolor": AQUA, "pad": 0.35}, arrowprops={"arrowstyle": "-", "color": AQUA, "lw": 0.9})
                ax.annotate(f"{label}\nPPV {ppv:.1f}%", xy=(xx, ppv), xytext=(88, 69), ha="center", fontsize=7.9, bbox={"facecolor": "white", "edgecolor": CORAL, "pad": 0.35}, arrowprops={"arrowstyle": "-", "color": CORAL, "lw": 0.9})
        ax.set(xlim=(0, 100), ylim=(0, 102), xlabel="Alert rate among eICU holdout sequences (%)", ylabel="Performance (%)")
        ax.set_title("Operating-point trade-offs", loc="left", fontsize=13.0, fontweight="bold", color=INK, pad=8)
        ax.grid(True, color="#D8D8D8", linewidth=0.6)
        ax.legend(loc="lower left", fontsize=8.0)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.87, bottom=0.16)
        save(fig, "Figure4B_operating_point_tradeoffs_horizontal")

    return SimpleNamespace(**locals())


_external = _load_external()


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Build Figure 4 cross-hospital transfer and operating-point panels (no options).")
        return
    _external.transfer()
    _external.transfer_narrow()
    _external.operating_points()


if __name__ == "__main__":
    main()
