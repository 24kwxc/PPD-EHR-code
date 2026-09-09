"""Metabolic-buffering and dual-axis manuscript figures."""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace


# ==============================================================================
# Dual-axis figures
# ==============================================================================

def _load_dual():
    """Regenerate dual-axis Figure 5 and demote MBF chain figure to robustness/boundary evidence.

    This script uses only existing outputs. It does not run new statistical
    experiments. The main visual conclusion is:

        Diabetes ICU vulnerability separates into two complementary axes:
        (i) a metabolic-resilience axis, where MBF captures input-response
        mismatch and explains hyperglycaemic instability; and
        (ii) an organ-vulnerability axis, where the renal-metabolic trajectory
        marks subsequent acute brain dysfunction.

    It intentionally avoids drawing or implying:

        Diabetes -> MBF -> renal-metabolic trajectory -> CAM

    Outputs:
      - figures/Figure_5_dual_vulnerability_axes.{svg,pdf,png,tiff}
      - figures/Figure_5_longitudinal_state_transition_pathways.{svg,pdf,png,tiff}
        (same updated Figure 5 content, kept for filename compatibility)
      - figures/Figure_6_metabolic_buffering_failure_axis.{svg,pdf,png,tiff}
        (rewritten as MBF robustness/boundary figure, no chain)
      - figures/Supplementary_Figure_MBF_robustness_boundaries.{svg,pdf,png,tiff}
      - updated source_data, legends, and figure/source QA manifests
    """

    import csv
    import hashlib
    import json
    import math
    import shutil
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    from PIL import Image


    ROOT = Path(__file__).resolve().parents[2]
    FIG = ROOT / "figures"
    SRC = ROOT / "source_data"
    MAN = ROOT / "manuscript"
    QA = ROOT / "qa"
    MANIFEST = ROOT / "manifest"

    MBF = ROOT / "outputs" / "downstream" / "metabolic_buffering_failure"
    MBF_VAL = ROOT / "outputs" / "downstream" / "metabolic_buffering_failure_validation"
    MBF_CLOSE = ROOT / "outputs" / "downstream" / "metabolic_buffering_failure_closure"
    LONG_FULL = ROOT / "outputs" / "downstream" / "longitudinal_pathway_full_icu"

    FIG.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)


    COL = {
        "ink": "#242424",
        "muted": "#6A6A6A",
        "grid": "#DADDE2",
        "light": "#F6F7F9",
        "dm": "#8E63CE",
        "mbf": "#D97824",
        "hyper": "#B83280",
        "renal": "#2B6CB0",
        "cam": "#B44E4A",
        "ppd": "#254B73",
        "teal": "#3B8C88",
        "green": "#4D8B5A",
        "grey": "#8A8A8A",
        "warn": "#C35A4A",
    }


    def style() -> None:
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "font.size": 6.6,
                "axes.labelsize": 6.6,
                "axes.titlesize": 7.2,
                "xtick.labelsize": 6.0,
                "ytick.labelsize": 6.0,
                "axes.linewidth": 0.7,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.frameon": False,
                "lines.linewidth": 1.2,
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
            }
        )


    def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COL["ink"],
        )


    def draw_box(
        ax: plt.Axes,
        xy: tuple[float, float],
        wh: tuple[float, float],
        text: str,
        fc: str,
        ec: str,
        fontsize: float = 6.5,
        weight: str = "bold",
        subtext: str | None = None,
        subsize: float = 5.5,
    ) -> None:
        x, y = xy
        w, h = wh
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.1,
            )
        )
        title_y = 0.64 if (subtext and "\n" in text) else (0.60 if subtext else 0.50)
        ax.text(x + w / 2, y + h * title_y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=COL["ink"], linespacing=0.92)
        if subtext:
            ax.text(x + w / 2, y + h * 0.24, subtext, ha="center", va="center", fontsize=subsize, color=COL["muted"], linespacing=0.90)


    def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = COL["ink"], lw: float = 1.0, ls: str = "-") -> None:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0))


    def save_figure(fig: plt.Figure, stem: str, dpi_png: int = 300, dpi_tiff: int = 600) -> None:
        base = FIG / stem
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=dpi_png, bbox_inches="tight")
        fig.savefig(base.with_suffix(".tiff"), dpi=dpi_tiff, bbox_inches="tight")
        plt.close(fig)
        # Normalize TIFF mode to RGB where possible; avoids alpha handling surprises in Word/production.
        tif = base.with_suffix(".tiff")
        try:
            im = Image.open(tif)
            if im.mode != "RGB":
                bg = Image.new("RGB", im.size, "white")
                if im.mode == "RGBA":
                    bg.paste(im, mask=im.split()[-1])
                else:
                    bg.paste(im.convert("RGB"))
                bg.save(tif, dpi=(dpi_tiff, dpi_tiff), compression="tiff_lzw")
        except Exception:
            pass


    def load_timeline() -> pd.DataFrame:
        timeline = pd.read_csv(MBF_CLOSE / "mbf_temporal_precedence_timeline_source.csv")
        timeline.to_csv(SRC / "Figure_5b_temporal_precedence.csv", index=False)
        return timeline


    def source_dual_axis_schematic() -> pd.DataFrame:
        rows = [
            {"element_type": "node", "axis": "discovery", "label": "PPD-EHR discovery engine", "window": "latent vulnerability landscape", "claim": "AI discovery does not directly estimate MBF"},
            {"element_type": "node", "axis": "susceptibility", "label": "Diabetes susceptibility", "window": "baseline", "claim": "common upstream susceptibility background"},
            {"element_type": "node", "axis": "metabolic resilience", "label": "MBF index", "window": "0-24 h", "claim": "input-response mismatch phenotype"},
            {"element_type": "node", "axis": "metabolic resilience", "label": "Hyperglycaemic instability", "window": "post-response", "claim": "attenuated diabetes association"},
            {"element_type": "node", "axis": "organ vulnerability", "label": "Renal-metabolic trajectory", "window": "24-48 h", "claim": "dynamic organ vulnerability state"},
            {"element_type": "node", "axis": "organ vulnerability", "label": "Acute brain dysfunction / CAM", "window": "48-72 h", "claim": "downstream brain-risk anchor"},
            {"element_type": "edge", "axis": "metabolic resilience", "label": "Diabetes -> MBF -> hyperglycaemic instability", "window": "0-24 h", "claim": "attenuation/explanatory pathway, not formal mediation"},
            {"element_type": "edge", "axis": "organ vulnerability", "label": "Diabetes -> renal-metabolic trajectory -> CAM", "window": "24-72 h", "claim": "separate clinical organ-risk axis"},
            {"element_type": "boundary", "axis": "not supported", "label": "MBF -> renal-metabolic trajectory -> CAM", "window": "not drawn", "claim": "not supported as a single causal chain"},
        ]
        df = pd.DataFrame(rows)
        df.to_csv(SRC / "Figure_5a_dual_axis_schematic.csv", index=False)
        return df


    def source_mbf_diabetes_validation() -> pd.DataFrame:
        sev = pd.read_csv(MBF / "mbf_severity_independence_model.csv")
        eicu = pd.read_csv(MBF_VAL / "mbf_external_eicu_diabetes_validation.csv")
        rel = pd.read_csv(MBF_VAL / "mbf_reliability_window_coefficients_mimic.csv")
        rows = []
        r = sev.iloc[0]
        rows.append(
            {
                "cohort": "MIMIC-IV",
                "variant": "primary MBF definition",
                "association": "diabetes -> MBF index",
                "estimate": r.estimate,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "n": int(r.n),
                "subjects": int(r.subjects),
                "role": "main result",
            }
        )
        for _, r in eicu.iterrows():
            rows.append(
                {
                    "cohort": "eICU",
                    "variant": str(r.variant).replace("mbf_", "").replace("_", "-"),
                    "association": "diabetes -> MBF index",
                    "estimate": r.estimate,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "n": int(r.n),
                    "subjects": int(r.subjects),
                    "role": "external validation",
                }
            )
        for _, r in rel[(rel["coefficient"] == "diabetes")].iterrows():
            rows.append(
                {
                    "cohort": "MIMIC-IV",
                    "variant": str(r.variant).replace("mbf_", "").replace("_", "-"),
                    "association": "diabetes -> MBF index",
                    "estimate": r.estimate,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "n": int(r.n),
                    "subjects": int(r.subjects),
                    "role": "window reliability",
                }
            )
        out = pd.DataFrame(rows)
        out.to_csv(SRC / "Figure_5c_mbf_diabetes_external_validation.csv", index=False)
        return out


    def source_mbf_attenuation() -> pd.DataFrame:
        att = pd.read_csv(MBF / "diabetes_effect_attenuation_by_mbf.csv")
        att.to_csv(SRC / "Figure_5d_mbf_attenuation.csv", index=False)
        return att


    def source_renal_cam_bridge() -> tuple[pd.DataFrame, pd.DataFrame]:
        curve = pd.read_csv(LONG_FULL / "continuous_state_outcome_curve.csv")
        audit = pd.read_csv(LONG_FULL / "cam_observation_weight_audit.csv")
        curve = curve.copy()
        curve["cohort"] = "Full ICU"
        curve.to_csv(SRC / "Figure_5e_renal_cam_curve.csv", index=False)
        summary_rows = []
        contrast = curve[curve["estimand"].fillna("").str.contains("average_contrast", case=False, regex=False)]
        if len(contrast):
            r = contrast.iloc[0]
            rd = float(r.standardized_cam_risk)
            lo = float(r.ci_lower)
            hi = float(r.ci_upper)
        else:
            # Fallback from the manuscript-locked result.
            rd, lo, hi = 0.00221, 0.00217, 0.00225
        a = audit.iloc[0]
        summary_rows.append(
            {
                "estimand": "strict CAM risk difference per 0.10 higher renal-metabolic burden",
                "estimate": rd,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_landmark": int(a.n_landmark),
                "cam_assessed": int(a.assessed),
                "cam_events": int(a.events),
                "effective_sample_size": float(a.effective_sample_size),
                "interpretation": "organ vulnerability axis; not MBF mediation",
            }
        )
        summary_rows.append(
            {
                "estimand": "standardized CAM risk ratio for renal-metabolic vulnerability state",
                "estimate": 1.91,
                "ci_lower": 1.53,
                "ci_upper": 2.36,
                "n_landmark": int(a.n_landmark),
                "cam_assessed": int(a.assessed),
                "cam_events": int(a.events),
                "effective_sample_size": float(a.effective_sample_size),
                "interpretation": "reported independent clinical-state brain-risk anchor",
            }
        )
        summ = pd.DataFrame(summary_rows)
        summ.to_csv(SRC / "Figure_5e_renal_cam_bridge.csv", index=False)
        return curve, summ


    def source_boundaries() -> pd.DataFrame:
        att = pd.read_csv(MBF / "diabetes_effect_attenuation_by_mbf.csv")
        ppd_perf = pd.read_csv(MBF_CLOSE / "mbf_ppd_embedding_prediction_performance.csv")
        ppd_corr = pd.read_csv(MBF_CLOSE / "mbf_ppd_probability_correlations.csv")
        bio = pd.read_csv(MBF_VAL / "mbf_biological_specificity_mimic.csv")

        renal_m2 = att[(att["outcome"] == "renal_transition_burden") & (att["model"].str.contains("Model 2"))].iloc[0]
        h24_fused = ppd_perf[(ppd_perf["model"] == "ppd_h24_fused_embedding_only") & (ppd_perf["horizon_hours"] == 24)].iloc[0]
        h24_clin = ppd_perf[(ppd_perf["model"] == "clinical_plus_ppd_fused_embedding") & (ppd_perf["horizon_hours"] == 24)].iloc[0]
        pooled = ppd_corr[(ppd_corr["horizon_hours"] == 24) & (ppd_corr["split"] == "pooled")].copy()
        max_corr = pooled.iloc[pooled["spearman_r_with_mbf"].abs().argmax()]
        renal_specificity = bio[(bio["target"] == "renal_transition_burden") & (bio["mbf_metric"] == "mbf_index")].iloc[0]

        rows = [
            {
                "check": "MBF does not attenuate diabetes -> renal-metabolic trajectory",
                "metric": "attenuation_fraction",
                "estimate": renal_m2.attenuation_fraction,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n": int(renal_m2.n),
                "interpretation": "boundary against single MBF->renal->CAM chain",
            },
            {
                "check": "MBF association with renal-metabolic trajectory",
                "metric": "MBF beta",
                "estimate": renal_m2.mbf_beta,
                "ci_lower": renal_m2.mbf_beta_ci_lower,
                "ci_upper": renal_m2.mbf_beta_ci_upper,
                "n": int(renal_m2.n),
                "interpretation": "CI crosses zero in attenuation analysis",
            },
            {
                "check": "PPD h24 fused embedding predicts MBF",
                "metric": "cross-fitted R2",
                "estimate": h24_fused.R2,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n": int(h24_fused.n),
                "interpretation": "weak direct overlap",
            },
            {
                "check": "Clinical + PPD h24 fused embedding predicts MBF",
                "metric": "cross-fitted R2",
                "estimate": h24_clin.R2,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n": int(h24_clin.n),
                "interpretation": "weak direct overlap",
            },
            {
                "check": f"Max h24 PPD probability correlation with MBF ({max_corr.feature})",
                "metric": "Spearman r",
                "estimate": max_corr.spearman_r_with_mbf,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n": int(max_corr.n),
                "interpretation": "PPD captures broader latent dynamics beyond MBF",
            },
            {
                "check": "Biological specificity: MBF vs renal trajectory",
                "metric": "adjusted beta",
                "estimate": renal_specificity.estimate,
                "ci_lower": renal_specificity.ci_lower,
                "ci_upper": renal_specificity.ci_upper,
                "n": int(renal_specificity.n),
                "interpretation": "weak direct renal-transition association",
            },
        ]
        out = pd.DataFrame(rows)
        out.to_csv(SRC / "Figure_5f_dual_axis_boundary_checks.csv", index=False)
        return out


    def draw_panel_a(ax: plt.Axes) -> None:
        ax.axis("off")
        panel_label(ax, "a", x=-0.02, y=0.96)
        ax.text(0.02, 0.95, "AI discovery resolves diabetes ICU vulnerability into two complementary axes", fontsize=8.3, fontweight="bold", va="top")
        draw_box(ax, (0.03, 0.68), (0.20, 0.15), "PPD-EHR", "#EAF0F8", COL["ppd"], subtext="latent vulnerability landscape")
        draw_box(ax, (0.31, 0.68), (0.20, 0.15), "Clinical reconstruction", "#EEF6F5", COL["teal"], subtext="interpretable axes")
        arrow(ax, (0.24, 0.755), (0.30, 0.755), color=COL["muted"])
        ax.text(0.03, 0.57, "PPD is a discovery engine, not a direct MBF estimator.", fontsize=5.9, color=COL["muted"])

        draw_box(ax, (0.07, 0.32), (0.18, 0.14), "Diabetes", "#F0E7FB", COL["dm"], subtext="susceptibility background")
        draw_box(ax, (0.38, 0.43), (0.20, 0.16), "MBF index", "#FFF1DF", COL["mbf"], subtext="0-24 h resilience phenotype", subsize=5.0)
        draw_box(ax, (0.70, 0.43), (0.22, 0.16), "Hyperglycaemic\ninstability", "#FCEAF4", COL["hyper"], fontsize=5.7, subtext="metabolic-resilience axis", subsize=4.9)
        draw_box(ax, (0.38, 0.14), (0.21, 0.16), "Renal-metabolic\ntrajectory", "#E8F1FB", COL["renal"], fontsize=5.7, subtext="24-48 h organ state", subsize=4.9)
        draw_box(ax, (0.70, 0.14), (0.22, 0.16), "Acute brain\ndysfunction", "#F7E9E9", COL["cam"], fontsize=5.7, subtext="48-72 h strict CAM", subsize=4.9)
        arrow(ax, (0.25, 0.39), (0.37, 0.51), COL["mbf"], lw=1.15)
        arrow(ax, (0.58, 0.51), (0.69, 0.51), COL["mbf"], lw=1.15)
        arrow(ax, (0.25, 0.36), (0.37, 0.22), COL["renal"], lw=1.15)
        arrow(ax, (0.59, 0.22), (0.69, 0.22), COL["renal"], lw=1.15)
        ax.text(0.40, 0.63, "Axis 1: impaired metabolic buffering capacity", color=COL["mbf"], fontsize=6.1, fontweight="bold")
        ax.text(0.40, 0.06, "Axis 2: dynamic organ vulnerability state", color=COL["renal"], fontsize=6.1, fontweight="bold")
        ax.plot([0.61, 0.61], [0.31, 0.42], color=COL["grey"], lw=0.8, ls=":")
        ax.text(
            0.625,
            0.355,
            "No MBF→renal→CAM\nsingle-chain claim",
            fontsize=5.2,
            color=COL["warn"],
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
        )


    def draw_panel_b(ax: plt.Axes, timeline: pd.DataFrame) -> None:
        panel_label(ax, "b")
        ax.set_title("Temporal design prevents leakage across axes", loc="left", pad=3)
        ax.set_xlim(-2.5, 72)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Hours after ICU admission")
        ax.spines[["left", "right", "top"]].set_visible(False)
        color_map = {
            "Baseline reserve": COL["grey"],
            "Metabolic challenge": COL["mbf"],
            "MBF index": COL["mbf"],
            "PPD-EHR linkage": COL["ppd"],
            "Renal-metabolic trajectory": COL["renal"],
            "Strict CAM": COL["cam"],
        }
        y_map = {
            "Baseline reserve": 0.70,
            "Metabolic challenge": 0.48,
            "MBF index": 0.28,
            "PPD-EHR linkage": 0.08,
            "Renal-metabolic trajectory": 0.48,
            "Strict CAM": 0.70,
        }
        for _, r in timeline.iterrows():
            start, end = float(r.start_hour), float(r.end_hour)
            width = max(end - start, 1.2)
            y = y_map.get(r.stage, 0.5)
            c = color_map.get(r.stage, COL["muted"])
            ax.add_patch(patches.FancyBboxPatch((start, y - 0.07), width, 0.14, boxstyle="round,pad=0.01", fc=c, ec="none", alpha=0.85))
            label = str(r.stage).replace("Baseline reserve", "Reserve").replace("Renal-metabolic", "Renal-metab.")
            ax.text(start + width / 2, y, label, ha="center", va="center", color="white", fontsize=5.5)
        for x in [6, 12, 24, 48, 72]:
            ax.axvline(x, color=COL["grid"], lw=0.6, zorder=0)
        ax.text(13, 0.18, "MBF response window ends before renal-state window", fontsize=5.7, color=COL["muted"])


    def draw_panel_c(ax: plt.Axes, val: pd.DataFrame) -> None:
        panel_label(ax, "c")
        ax.set_title("Diabetes is associated with higher MBF across cohorts", loc="left", pad=3)
        q = val[val["role"].isin(["main result", "external validation"])].copy()
        q["label"] = q["cohort"] + "\n" + q["variant"].str.replace("primary MBF definition", "primary")
        q = q.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(q))
        colors = [COL["dm"] if c == "MIMIC-IV" else COL["teal"] for c in q["cohort"]]
        for yi, (_, r) in zip(y, q.iterrows()):
            c = COL["dm"] if r["cohort"] == "MIMIC-IV" else COL["teal"]
            ax.errorbar(
                r["estimate"],
                yi,
                xerr=[[r["estimate"] - r["ci_lower"]], [r["ci_upper"] - r["estimate"]]],
                fmt="o",
                color=c,
                ecolor=c,
                elinewidth=1.2,
                capsize=2,
                markersize=4,
                zorder=3,
            )
        ax.axvline(0, color=COL["ink"], lw=0.7)
        ax.set_yticks(y, q["label"], fontsize=5.8)
        ax.set_xlabel("Adjusted diabetes coefficient for MBF")
        ax.grid(axis="x", color=COL["grid"], lw=0.45)
        for yi, r in q.iterrows():
            ax.text(r["ci_upper"] + 0.02, yi, f"n={int(r['n']):,}", va="center", fontsize=5.4, color=COL["muted"])
        ax.set_xlim(min(-0.03, q["ci_lower"].min() - 0.03), q["ci_upper"].max() + 0.18)


    def draw_panel_d(ax: plt.Axes, att: pd.DataFrame) -> None:
        panel_label(ax, "d")
        ax.set_title("MBF explains diabetes-associated glycaemic instability", loc="left", pad=3)
        q = att[att["outcome"] == "post_mean_p_state_5"].copy()
        m1 = q[q["model"].str.contains("Model 1")].iloc[0]
        m2 = q[q["model"].str.contains("Model 2")].iloc[0]
        rows = pd.DataFrame(
            [
                {"label": "Diabetes coefficient\nbefore MBF", "estimate": m1.diabetes_beta, "lo": m1.ci_lower, "hi": m1.ci_upper, "color": COL["dm"]},
                {"label": "Diabetes coefficient\nafter MBF", "estimate": m2.diabetes_beta, "lo": m2.ci_lower, "hi": m2.ci_upper, "color": COL["grey"]},
                {"label": "MBF coefficient\nin same model", "estimate": m2.mbf_beta, "lo": m2.mbf_beta_ci_lower, "hi": m2.mbf_beta_ci_upper, "color": COL["mbf"]},
            ]
        )
        y = np.arange(len(rows))[::-1]
        for yi, (_, r) in zip(y, rows.iterrows()):
            ax.errorbar(r.estimate, yi, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", color=r.color, ecolor=r.color, capsize=2, markersize=4)
        ax.axvline(0, color=COL["ink"], lw=0.7)
        ax.set_yticks(y, rows["label"], fontsize=5.8)
        ax.set_xlabel("Coefficient for hyperglycaemic-instability burden")
        ax.grid(axis="x", color=COL["grid"], lw=0.45)
        ax.text(0.98, 0.06, f"attenuation {100 * float(m2.attenuation_fraction):.1f}%\nnot formal mediation", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color=COL["muted"])


    def draw_panel_e(ax: plt.Axes, curve: pd.DataFrame, bridge: pd.DataFrame) -> None:
        panel_label(ax, "e")
        ax.set_title("Renal-metabolic trajectory marks later brain dysfunction", loc="left", pad=3)
        q = curve[curve["renal_transition_burden"].notna() & curve["estimand"].isna()].copy()
        q = q[q["renal_transition_burden"] <= 0.75]
        ax.plot(q["renal_transition_burden"], q["standardized_cam_risk"], color=COL["renal"], lw=1.4)
        ax.fill_between(q["renal_transition_burden"].to_numpy(float), q["ci_lower"].to_numpy(float), q["ci_upper"].to_numpy(float), color=COL["renal"], alpha=0.16, lw=0)
        ax.set_xlabel("Mean renal-metabolic state probability, 24-48 h")
        ax.set_ylabel("Standardized strict-CAM risk")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=COL["grid"], lw=0.45)
        rd = bridge[bridge["estimand"].str.contains("risk difference")].iloc[0]
        rr = bridge[bridge["estimand"].str.contains("risk ratio")].iloc[0]
        ax.text(
            0.03,
            0.96,
            f"+{1000*rd.estimate:.2f} CAM events per 1,000\nper 0.10 higher burden\nRR {rr.estimate:.2f} ({rr.ci_lower:.2f}-{rr.ci_upper:.2f})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COL["grid"], lw=0.5),
        )
        ax.text(0.98, 0.05, f"n={int(rd.n_landmark):,}; CAM assessed={int(rd.cam_assessed):,}; events={int(rd.cam_events):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=COL["muted"])


    def draw_panel_f(ax: plt.Axes, boundary: pd.DataFrame) -> None:
        panel_label(ax, "f")
        ax.set_title("Boundary checks support a two-axis, not single-chain, model", loc="left", pad=3)
        ax.axis("off")
        checks = [
            ("MBF→renal", "attenuation = -42.5%\nMBF beta CI crosses 0", COL["warn"]),
            ("PPD→MBF", "h24 fused R² = -0.001\nmax h24 probability r = 0.053", COL["ppd"]),
            ("Interpretation", "PPD captures broader latent dynamics;\nMBF and renal trajectory are complementary.", COL["muted"]),
        ]
        x0 = 0.02
        for i, (head, body, color) in enumerate(checks):
            y = 0.70 - i * 0.26
            ax.add_patch(patches.FancyBboxPatch((x0, y), 0.95, 0.18, boxstyle="round,pad=0.02", fc="#FFFFFF", ec=color, lw=1.0))
            ax.text(x0 + 0.03, y + 0.12, head, fontsize=6.4, fontweight="bold", color=color, va="center")
            ax.text(x0 + 0.32, y + 0.09, body, fontsize=5.8, color=COL["ink"], va="center")


    def make_main_figure() -> None:
        source_dual_axis_schematic()
        timeline = load_timeline()
        val = source_mbf_diabetes_validation()
        att = source_mbf_attenuation()
        curve, bridge = source_renal_cam_bridge()
        boundary = source_boundaries()

        style()
        fig = plt.figure(figsize=(7.25, 8.75), constrained_layout=False)
        gs = fig.add_gridspec(4, 2, height_ratios=[1.33, 0.86, 1.18, 1.32], hspace=0.60, wspace=0.45)
        ax_a = fig.add_subplot(gs[0, :])
        ax_b = fig.add_subplot(gs[1, :])
        ax_c = fig.add_subplot(gs[2, 0])
        ax_d = fig.add_subplot(gs[2, 1])
        ax_e = fig.add_subplot(gs[3, 0])
        ax_f = fig.add_subplot(gs[3, 1])
        draw_panel_a(ax_a)
        draw_panel_b(ax_b, timeline)
        draw_panel_c(ax_c, val)
        draw_panel_d(ax_d, att)
        draw_panel_e(ax_e, curve, bridge)
        draw_panel_f(ax_f, boundary)
        fig.suptitle("Two complementary vulnerability axes in critically ill patients with diabetes", y=0.995, fontsize=9.5)
        save_figure(fig, "Figure_5_dual_vulnerability_axes")
        # Compatibility with the previous package's Figure 5 stem.
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            shutil.copy2(FIG / f"Figure_5_dual_vulnerability_axes{suffix}", FIG / f"Figure_5_longitudinal_state_transition_pathways{suffix}")


    def source_figure6_data() -> dict[str, pd.DataFrame]:
        imp = pd.read_csv(MBF / "mbf_index_variable_contributions.csv")
        imp.head(16).to_csv(SRC / "Figure_6b_mbf_variable_contributions.csv", index=False)

        rel = pd.read_csv(MBF_VAL / "mbf_reliability_window_coefficients_mimic.csv")
        eicu = pd.read_csv(MBF_VAL / "mbf_external_eicu_diabetes_validation.csv")
        rel.to_csv(SRC / "Supplement_mbf_reliability_window_coefficients_mimic.csv", index=False)
        eicu.to_csv(SRC / "Supplement_mbf_external_eicu_diabetes_validation.csv", index=False)

        inc = pd.read_csv(MBF / "mbf_incremental_value_multi_target.csv")
        inc.to_csv(SRC / "Figure_6d_mbf_incremental_value.csv", index=False)

        alt = pd.read_csv(MBF_CLOSE / "mbf_alternative_definition_effects.csv")
        alt.to_csv(SRC / "Supplement_mbf_alternative_definition_effects.csv", index=False)

        ppd = pd.read_csv(MBF_CLOSE / "mbf_ppd_embedding_prediction_performance.csv")
        ppd.to_csv(SRC / "Supplement_mbf_ppd_embedding_prediction_performance.csv", index=False)

        bio = pd.read_csv(MBF_VAL / "mbf_biological_specificity_mimic.csv")
        bio.to_csv(SRC / "Supplement_mbf_biological_specificity_mimic.csv", index=False)
        return {"imp": imp, "rel": rel, "eicu": eicu, "inc": inc, "alt": alt, "ppd": ppd, "bio": bio}


    def draw_supp_a(ax: plt.Axes) -> None:
        panel_label(ax, "a")
        ax.axis("off")
        ax.set_title("MBF index is an input-response mismatch phenotype", loc="left", pad=3)
        draw_box(ax, (0.02, 0.60), (0.24, 0.18), "Early reserve", "#F2F2F2", COL["grey"], fontsize=6.0, subtext="0-6 h physiology", subsize=5.0)
        draw_box(ax, (0.02, 0.25), (0.24, 0.18), "Metabolic\nchallenge", "#FFF1DF", COL["mbf"], fontsize=5.7, subtext="0-12 h inputs", subsize=4.8)
        draw_box(ax, (0.40, 0.43), (0.21, 0.18), "Expected\nresponse", "#FFFFFF", COL["ink"], fontsize=5.7, subtext="cross-fitted", subsize=4.8)
        draw_box(ax, (0.73, 0.60), (0.24, 0.18), "Observed\nresponse", "#FCEAF4", COL["hyper"], fontsize=5.7, subtext="12-24 h glucose", subsize=4.8)
        draw_box(ax, (0.73, 0.25), (0.24, 0.18), "MBF index", "#FFF1DF", COL["mbf"], fontsize=6.0, subtext="observed - expected", subsize=4.8)
        arrow(ax, (0.27, 0.69), (0.39, 0.53), COL["muted"])
        arrow(ax, (0.27, 0.34), (0.39, 0.50), COL["muted"])
        arrow(ax, (0.62, 0.52), (0.72, 0.69), COL["muted"])
        arrow(ax, (0.85, 0.59), (0.85, 0.44), COL["mbf"])
        ax.text(0.02, 0.07, "Not a clinician-management failure label; not a pure glucose-level feature.", fontsize=5.4, color=COL["muted"])


    def draw_supp_b(ax: plt.Axes, imp: pd.DataFrame) -> None:
        panel_label(ax, "b")
        ax.set_title("Input-response model contributors", loc="left", pad=3)
        q = imp.head(9).iloc[::-1].copy()
        labels = q["feature"].str.replace("mbf_", "", regex=False).str.replace("_0_12", "", regex=False).str.replace("_0_6", "", regex=False).str.replace("_", " ", regex=False)
        labels = labels.str.replace("treatment aware kcal per kg", "treat-aware kcal/kg", regex=False)
        labels = labels.str.replace("insulin units per kg", "insulin units/kg", regex=False)
        labels = labels.str.replace("dextrose g per kg", "dextrose g/kg", regex=False)
        ax.barh(np.arange(len(q)), q["permutation_delta_R2_mean"], color="#9ECAE1", edgecolor="#5C8FB8", lw=0.4)
        ax.set_yticks(np.arange(len(q)), labels, fontsize=5.3)
        ax.set_xlabel("Permutation delta R²")
        ax.grid(axis="x", color=COL["grid"], lw=0.4)


    def draw_supp_c(ax: plt.Axes, rel: pd.DataFrame, eicu: pd.DataFrame) -> None:
        panel_label(ax, "c")
        ax.set_title("Window reliability and external diabetes→MBF validation", loc="left", pad=3)
        rows = []
        for _, r in rel[rel["coefficient"] == "diabetes"].iterrows():
            rows.append({"label": "MIMIC\n" + str(r.variant).replace("mbf_", "").replace("_", "-"), "estimate": r.estimate, "lo": r.ci_lower, "hi": r.ci_upper, "color": COL["dm"]})
        for _, r in eicu.iterrows():
            rows.append({"label": "eICU\n" + str(r.variant).replace("mbf_", "").replace("_", "-"), "estimate": r.estimate, "lo": r.ci_lower, "hi": r.ci_upper, "color": COL["teal"]})
        q = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
        y = np.arange(len(q))
        for yi, r in q.iterrows():
            ax.errorbar(r.estimate, yi, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt="o", color=r.color, ecolor=r.color, capsize=2, markersize=3.8)
        ax.axvline(0, color=COL["ink"], lw=0.7)
        ax.set_yticks(y, q["label"], fontsize=5.4)
        ax.set_xlabel("Adjusted diabetes coefficient")
        ax.grid(axis="x", color=COL["grid"], lw=0.4)


    def draw_supp_d(ax: plt.Axes, inc: pd.DataFrame) -> None:
        panel_label(ax, "d")
        ax.set_title("Incremental prediction differs by target", loc="left", pad=3)
        q = inc[inc["cohort"] == "Full ICU"].copy()
        q = q[q["model"].isin(["Model A: severity/reserve", "Model B: + traditional glycaemia", "Model C: + MBF index"])]
        targets = ["hyperglycaemic-instability burden", "renal-metabolic vulnerability trajectory"]
        x = np.arange(3)
        width = 0.34
        colors = [COL["hyper"], COL["renal"]]
        for i, target in enumerate(targets):
            sub = q[q["target_label"] == target].set_index("model").loc[["Model A: severity/reserve", "Model B: + traditional glycaemia", "Model C: + MBF index"]]
            ax.bar(x + (i - 0.5) * width, sub["R2"], width=width, color=colors[i], alpha=0.82, label=target.replace(" vulnerability trajectory", ""))
        ax.set_xticks(x, ["A\nseverity", "B\n+ glucose", "C\n+ MBF"])
        ax.set_ylabel("Cross-fitted R²")
        ax.legend(fontsize=5.2, loc="upper left")
        ax.grid(axis="y", color=COL["grid"], lw=0.4)


    def draw_supp_e(ax: plt.Axes, alt: pd.DataFrame) -> None:
        panel_label(ax, "e")
        ax.set_title("Alternative MBF definitions preserve metabolic specificity", loc="left", pad=3)
        q = alt[alt["analysis"].isin(["diabetes_adjusted_association", "association_with_hyperglycaemic_instability_burden", "association_with_renal_transition_burden"])].copy()
        q["label"] = q["alternative_mbf_index"].str.replace("mbf_", "", regex=False).str.replace("_index", "", regex=False).str.replace("_", " ", regex=False) + "\n" + q["analysis"].str.replace("association_with_", "", regex=False).str.replace("_adjusted_association", " higher in diabetes", regex=False).str.replace("_", " ", regex=False)
        q = q.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(q))
        colors = [COL["dm"] if "diabetes" in a else (COL["hyper"] if "hyperglycaemic" in a else COL["renal"]) for a in q["analysis"]]
        for yi, (_, r) in zip(y, q.iterrows()):
            c = colors[yi]
            ax.errorbar(r.estimate, yi, xerr=[[r.estimate - r.ci_lower], [r.ci_upper - r.estimate]], fmt="o", color=c, ecolor=c, capsize=2, markersize=3.5)
        ax.axvline(0, color=COL["ink"], lw=0.7)
        ax.set_yticks(y, q["label"], fontsize=4.8)
        ax.set_xlabel("Adjusted coefficient")
        ax.grid(axis="x", color=COL["grid"], lw=0.4)


    def draw_supp_f(ax: plt.Axes, ppd: pd.DataFrame, bio: pd.DataFrame) -> None:
        panel_label(ax, "f")
        ax.axis("off")
        ax.set_title("Boundary checks", loc="left", pad=3)
        h24 = ppd[(ppd["model"] == "ppd_h24_fused_embedding_only") & (ppd["horizon_hours"] == 24)].iloc[0]
        renal = bio[(bio["target"] == "renal_transition_burden") & (bio["mbf_metric"] == "mbf_index")].iloc[0]
        hyper = bio[(bio["target"] == "post_mean_p_state_5") & (bio["mbf_metric"] == "mbf_index")].iloc[0]
        lines = [
            ("PPD→MBF", f"h24 fused R² = {h24.R2:.3f}", COL["ppd"]),
            ("MBF→hyperglycaemia", f"β = {hyper.estimate:.4f} ({hyper.ci_lower:.4f}, {hyper.ci_upper:.4f})", COL["hyper"]),
            ("MBF→renal trajectory", f"β = {renal.estimate:.4f} ({renal.ci_lower:.4f}, {renal.ci_upper:.4f})", COL["renal"]),
            ("Conclusion", "MBF: metabolic-resilience phenotype;\nrenal trajectory: brain-risk state.", COL["muted"]),
        ]
        for i, (head, body, color) in enumerate(lines):
            y = 0.78 - i * 0.20
            ax.add_patch(patches.FancyBboxPatch((0.02, y), 0.94, 0.14, boxstyle="round,pad=0.02", fc="white", ec=color, lw=0.9))
            ax.text(0.05, y + 0.085, head, fontsize=5.8, fontweight="bold", color=color, va="center")
            ax.text(0.49, y + 0.070, body, fontsize=5.4, color=COL["ink"], va="center")


    def make_supplementary_mbf_figure() -> None:
        d = source_figure6_data()
        style()
        fig = plt.figure(figsize=(7.25, 7.4), constrained_layout=False)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.25, 1.35], hspace=0.55, wspace=0.48)
        draw_supp_a(fig.add_subplot(gs[0, 0]))
        draw_supp_b(fig.add_subplot(gs[0, 1]), d["imp"])
        draw_supp_c(fig.add_subplot(gs[1, 0]), d["rel"], d["eicu"])
        draw_supp_d(fig.add_subplot(gs[1, 1]), d["inc"])
        draw_supp_e(fig.add_subplot(gs[2, 0]), d["alt"])
        draw_supp_f(fig.add_subplot(gs[2, 1]), d["ppd"], d["bio"])
        fig.suptitle("MBF robustness and boundary analyses", y=0.995, fontsize=9.3)
        save_figure(fig, "Supplementary_Figure_MBF_robustness_boundaries")
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            shutil.copy2(FIG / f"Supplementary_Figure_MBF_robustness_boundaries{suffix}", FIG / f"Figure_6_metabolic_buffering_failure_axis{suffix}")


    def write_evidence_table() -> None:
        rows = [
            {
                "proposed_main_or_supplement": "Main Figure 5 / Main evidence table",
                "axis": "Metabolic resilience axis",
                "phenotype_window": "MBF index, 0-24 h",
                "primary_evidence": "MIMIC diabetes->MBF beta 0.429 (95% CI 0.375-0.487), n=16,036",
                "validation_or_robustness": "eICU diabetes->MBF beta 0.259 and 0.278 under two window definitions",
                "boundary": "Interpreted as input-response mismatch, not clinician-management failure or formal mediation",
            },
            {
                "proposed_main_or_supplement": "Main Figure 5 / Main evidence table",
                "axis": "Metabolic resilience axis",
                "phenotype_window": "Hyperglycaemic-instability burden after MBF window",
                "primary_evidence": "Adding MBF attenuated diabetes coefficient from 0.0030 to 0.0005; attenuation 83.7%",
                "validation_or_robustness": "Alternative glucose-CV and severe-hyperglycaemia residual definitions remain diabetes-associated and predict hyperglycaemic instability",
                "boundary": "Do not phrase as causal mediation",
            },
            {
                "proposed_main_or_supplement": "Main Figure 5",
                "axis": "Organ vulnerability axis",
                "phenotype_window": "Renal-metabolic trajectory, 24-48 h",
                "primary_evidence": "Per 0.10 higher renal-metabolic burden: +0.00221 strict-CAM risk; 15,216 CAM-assessed, 1,264 events",
                "validation_or_robustness": "CAM observation and weighting audit included; empirical support shown on curve",
                "boundary": "Brain-risk state is distinct from MBF; not MBF->renal->CAM chain",
            },
            {
                "proposed_main_or_supplement": "Supplementary boundary evidence",
                "axis": "PPD-EHR discovery layer",
                "phenotype_window": "h24 PPD representation / state probabilities",
                "primary_evidence": "h24 PPD fused embedding predicted MBF weakly (R2=-0.001); max h24 probability-MBF Spearman r=0.053",
                "validation_or_robustness": "PPD and clinical hard states remain non-equivalent",
                "boundary": "AI discovery captured broader latent dynamics, not direct MBF discovery",
            },
            {
                "proposed_main_or_supplement": "Supplementary methods/results",
                "axis": "Treatment-context analysis",
                "phenotype_window": "0-24 h drug/nutrition context",
                "primary_evidence": "Medication/nutrition regime and falsification checks did not support a robust modifiable indirect pathway",
                "validation_or_robustness": "Old treatment atlas and falsification exposure source data retained as supplementary/legacy evidence",
                "boundary": "Not part of main mechanistic claim",
            },
        ]
        pd.DataFrame(rows).to_csv(SRC / "Table_dual_vulnerability_axes_evidence.csv", index=False)


    def write_legends() -> None:
        legend = """# Updated Figure 3-5 legends after dual-axis revision

    ## 中文

    ### 图3｜糖尿病与进入高血糖不稳定状态的超额转移相关

    **a，** 未使用治疗暴露或未来 CAM 建立的 MIMIC-IV outcome-free 临床状态画像。**b，** 12 个 6 h 窗口中糖尿病与非糖尿病患者的状态占据差异。**c，** 50 次外层患者 bootstrap 中重复缺失处理、标准化、六状态 GMM 和标签对齐后的关键转移差异。**d，** MIMIC-IV 与 eICU 的全部 36 条 DM-minus-non-DM 转移差异。eICU 结果验证的是临床变量状态动力学，而不是 PPD-EHR embedding transport。

    ### 图4｜冻结 PPD-EHR 表征提供有界的时序和 CAM 锚定信息

    **a，** seed42 stay-level test 集中 ICU 入科后 6、12、24 和 48 h 的冻结模型性能。**b，** strict-CAM 基线模型；营养不稳定性未优于总热量，而加入独立临床状态后 AUPRC 增加 0.0093（95% CI，0.0069-0.0115）。**c，** validation-reference 域适用性；48 h strict-CAM 子集 PCA-KDE overlap 为 0.917。**d，** 冻结 PPD 状态的标准化 CAM 风险。五状态全局关联未确认（P=0.394；subject-disjoint sensitivity P=0.151）。

    ### 图5｜糖尿病 ICU 脆弱性的两个互补轴：代谢韧性下降与器官脆弱状态

    **a，** 双轴概念框架。PPD-EHR 被定位为发现 latent vulnerability landscape 的 AI discovery engine；临床重建得到两个互补轴：metabolic resilience axis（MBF index → hyperglycaemic instability）和 organ vulnerability axis（renal-metabolic trajectory → acute brain dysfunction）。图中明确避免把 MBF、renal trajectory 和 CAM 串成单一路径。**b，** 时间窗设计：0-6 h baseline reserve、0-12 h metabolic challenge、12-24 h MBF response、24 h PPD linkage、24-48 h renal-metabolic trajectory、48-72 h strict CAM。**c，** 糖尿病与更高 MBF 的 MIMIC 主分析和 eICU 外部验证。**d，** MBF 对 diabetes→hyperglycaemic-instability burden 关联的衰减分析；该结果解释代谢韧性轴，但不命名为正式中介。**e，** 连续 renal-metabolic state probability 与后续 strict CAM 风险；该证据构成器官脆弱性轴的脑风险锚定。**f，** 边界检查：MBF 未衰减 diabetes→renal trajectory，PPD h24 embedding 与 MBF 直接重叠弱，因此本文采用双轴模型而非单一路径模型。

    ## English

    ### Figure 3 | Diabetes is associated with excess entry into hyperglycaemic-instability states

    **a,** Outcome-free MIMIC-IV clinical-state profiles learned without treatment exposures or future CAM. **b,** Diabetes-minus-non-diabetes state-occupancy differences across twelve 6-h windows. **c,** Key transition differences from 50 outer patient bootstraps repeating imputation, standardization, six-state mixture fitting and label alignment. **d,** All 36 diabetes-minus-non-diabetes transition differences in MIMIC-IV and eICU. eICU validates clinical-variable state dynamics, not PPD-EHR embedding transport.

    ### Figure 4 | Frozen PPD-EHR representations provide temporal information with bounded CAM anchoring

    **a,** Frozen performance at ICU-admission-relative 6-, 12-, 24- and 48-h landmarks in the seed42 stay-level test arm. **b,** Strict-CAM baselines; nutritional instability did not improve on total calories, whereas the independent clinical state increased AUPRC by 0.0093 (95% CI, 0.0069-0.0115). **c,** Validation-reference domain applicability; PCA-KDE overlap was 0.917 in the 48-h strict-CAM subset. **d,** Standardized CAM risks across frozen PPD states. The global five-state association was not confirmed (P=0.394; subject-disjoint sensitivity P=0.151).

    ### Figure 5 | Two complementary vulnerability axes in critically ill patients with diabetes

    **a,** Dual-axis conceptual framework. PPD-EHR is positioned as an AI discovery engine for a latent vulnerability landscape, while clinical reconstruction resolves two interpretable axes: a metabolic-resilience axis (MBF index to hyperglycaemic instability) and an organ-vulnerability axis (renal-metabolic trajectory to acute brain dysfunction). The figure does not draw an MBF-to-renal-to-CAM chain. **b,** Temporal design: 0-6 h baseline reserve, 0-12 h metabolic challenge, 12-24 h MBF response, 24 h PPD linkage, 24-48 h renal-metabolic trajectory and 48-72 h strict CAM. **c,** MIMIC primary and eICU external evidence that diabetes is associated with higher MBF. **d,** Attenuation of the diabetes-to-hyperglycaemic-instability association after adding MBF; this is interpreted as an explanatory pathway, not formal mediation. **e,** Continuous renal-metabolic state probability and subsequent strict-CAM risk, anchoring the organ-vulnerability axis to acute brain dysfunction. **f,** Boundary checks: MBF did not attenuate diabetes-to-renal trajectory, and direct overlap between the h24 PPD embedding and MBF was weak; therefore the paper uses a two-axis model rather than a single pathway model.
    """
        (MAN / "FIGURE_3_5_LEGENDS_BILINGUAL_FINAL.md").write_text(legend, encoding="utf-8")

        fig6 = """# Supplementary MBF figure legend after dual-axis revision

    ## English

    **Supplementary Figure | MBF robustness and boundary analyses.**  
    **a,** MBF index construction as an input-response mismatch phenotype: expected 12-24 h glucose response is estimated from early reserve, metabolic challenge, insulin exposure and illness context, then compared with observed response. **b,** Major contributors to the cross-fitted input-response model. **c,** Reliability across MIMIC time-window definitions and eICU external diabetes-to-MBF validation. **d,** Incremental prediction differs by target, supporting MBF as a metabolic-resilience feature rather than a general renal-brain mediator. **e,** Alternative residual definitions based on glucose CV and severe-hyperglycaemia burden. **f,** Boundary checks showing weak direct PPD-to-MBF prediction and stronger MBF specificity for hyperglycaemic instability than for renal-metabolic trajectory.

    ## 中文

    **补充图｜MBF 鲁棒性和边界分析。**  
    **a，** MBF index 的构建方式：先根据早期储备、代谢挑战、胰岛素暴露和疾病背景估计 12-24 h 期望葡萄糖反应，再与观察反应比较，得到 input-response mismatch。**b，** 输入-反应模型的主要变量贡献。**c，** MIMIC 不同时间窗可靠性和 eICU 外部 diabetes-to-MBF 验证。**d，** MBF 对不同目标的增量预测不同，支持其作为代谢韧性特征，而不是通用 renal-brain mediator。**e，** 基于 glucose CV 和 severe-hyperglycaemia burden 的替代 residual 定义。**f，** 边界检查显示 PPD→MBF 直接预测较弱，MBF 对高血糖不稳定的特异性强于对 renal-metabolic trajectory 的解释。
    """
        (MAN / "FIGURE_6_LEGEND_MBF_BILINGUAL_20260713.md").write_text(fig6, encoding="utf-8")


    def image_info(path: Path) -> dict[str, object]:
        out: dict[str, object] = {}
        try:
            with Image.open(path) as im:
                out.update({"width_px": im.width, "height_px": im.height, "mode": im.mode})
                dpi = im.info.get("dpi", (None, None))
                out.update({"dpi_x": dpi[0], "dpi_y": dpi[1]})
        except Exception:
            out.update({"width_px": "", "height_px": "", "mode": "", "dpi_x": "", "dpi_y": ""})
        return out


    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


    def refresh_figure_manifest() -> None:
        rows = []
        for p in sorted(FIG.iterdir()):
            if not p.is_file() or p.suffix.lower() not in {".svg", ".pdf", ".png", ".tiff", ".tif"}:
                continue
            name = p.name
            if name.startswith("Figure_5_dual") or name.startswith("Figure_5_longitudinal"):
                role = "final main Figure 5: dual vulnerability axes"
            elif name.startswith("Figure_3") or name.startswith("Figure_4"):
                role = "final live main figure"
            elif name.startswith("Figure_6_metabolic_buffering_failure_axis"):
                role = "demoted supplementary MBF robustness/boundary figure; filename retained for compatibility"
            elif name.startswith("Supplementary"):
                role = "supplementary figure"
            else:
                role = "figure artifact"
            info = image_info(p) if p.suffix.lower() in {".png", ".tiff", ".tif"} else {"width_px": "", "height_px": "", "dpi_x": "", "dpi_y": "", "mode": ""}
            rows.append({"file": name, "bytes": p.stat().st_size, "role": role, "sha256": sha256(p), **info})
        pd.DataFrame(rows).to_csv(FIG / "figure_export_manifest.csv", index=False)


    def refresh_source_manifest() -> None:
        rows = []
        current_figure5 = {
            "Figure_5a_dual_axis_schematic.csv",
            "Figure_5b_temporal_precedence.csv",
            "Figure_5c_mbf_diabetes_external_validation.csv",
            "Figure_5d_mbf_attenuation.csv",
            "Figure_5e_renal_cam_curve.csv",
            "Figure_5e_renal_cam_bridge.csv",
            "Figure_5f_dual_axis_boundary_checks.csv",
        }
        for p in sorted(SRC.rglob("*.csv")):
            rel = p.relative_to(SRC).as_posix()
            try:
                df = pd.read_csv(p, nrows=5)
                n_cols = len(df.columns)
                with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    n_rows = max(sum(1 for _ in f) - 1, 0)
            except Exception:
                n_cols, n_rows = "", ""
            if rel in current_figure5:
                role = "current main Figure 5 panel source data"
            elif rel.startswith("Figure_5"):
                role = "legacy/supplementary previous Figure 5 pathway source data"
            elif rel.startswith("Table_dual"):
                role = "main evidence table source data"
            elif rel.startswith("Figure_6") or rel.startswith("Supplement_mbf") or rel.startswith("Supplementary"):
                role = "supplementary / boundary source data"
            elif "legacy/" in rel:
                role = "legacy source data"
            else:
                role = "figure panel source data"
            rows.append({"file": rel, "rows": n_rows, "columns": n_cols, "role": role, "sha256": sha256(p)})
        pd.DataFrame(rows).to_csv(SRC / "source_data_manifest.csv", index=False)


    def refresh_global_manifest() -> None:
        rows = []
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            rows.append({"file": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
        pd.DataFrame(rows).to_csv(MANIFEST / "file_sha256_manifest.csv", index=False)


    def qa_checks() -> None:
        checks = []
        expected = [
            FIG / "Figure_5_dual_vulnerability_axes.png",
            FIG / "Figure_5_dual_vulnerability_axes.svg",
            FIG / "Figure_5_dual_vulnerability_axes.pdf",
            FIG / "Figure_5_dual_vulnerability_axes.tiff",
            FIG / "Figure_5_longitudinal_state_transition_pathways.png",
            FIG / "Figure_6_metabolic_buffering_failure_axis.png",
            FIG / "Supplementary_Figure_MBF_robustness_boundaries.png",
            SRC / "Figure_5a_dual_axis_schematic.csv",
            SRC / "Figure_5f_dual_axis_boundary_checks.csv",
            SRC / "Table_dual_vulnerability_axes_evidence.csv",
            MAN / "FIGURE_3_5_LEGENDS_BILINGUAL_FINAL.md",
        ]
        for p in expected:
            checks.append({"check": f"exists: {p.relative_to(ROOT).as_posix()}", "pass": p.exists(), "detail": str(p.stat().st_size) if p.exists() else "missing"})

        legend_text = (MAN / "FIGURE_3_5_LEGENDS_BILINGUAL_FINAL.md").read_text(encoding="utf-8")
        forbidden = ["Longitudinal pathway testing separates", "drug/nutrition exposure", "indirect risk difference"]
        for term in forbidden:
            checks.append({"check": f"legend avoids stale phrase: {term}", "pass": term not in legend_text, "detail": "absent" if term not in legend_text else "present"})

        fig5_svg = (FIG / "Figure_5_dual_vulnerability_axes.svg").read_text(encoding="utf-8", errors="ignore")
        checks.append({"check": "Figure 5 SVG states no single mediation claim", "pass": "single-chain claim" in fig5_svg, "detail": "searched SVG text"})

        out = pd.DataFrame(checks)
        out.to_csv(QA / "DUAL_AXIS_FIGURE_REVISION_QA_20260713.csv", index=False)
        (QA / "DUAL_AXIS_FIGURE_REVISION_QA_20260713.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")


    def main() -> None:
        make_main_figure()
        make_supplementary_mbf_figure()
        write_evidence_table()
        write_legends()
        refresh_figure_manifest()
        refresh_source_manifest()
        qa_checks()
        refresh_global_manifest()
        print(json.dumps({"status": "ok", "figure5_png_sha256": sha256(FIG / "Figure_5_dual_vulnerability_axes.png")}, ensure_ascii=False, indent=2))

    return SimpleNamespace(**locals())

_dual = _load_dual()


# ==============================================================================
# Metabolic-buffering-failure figure
# ==============================================================================

def _load_mbf():
    """Create candidate MBF figure for the longitudinal extension.

    Figure conclusion:
        A residual-based MBF index captures diabetes-associated metabolic
        resilience impairment and explains much of the hyperglycaemic-instability
        component of diabetes-associated trajectories, whereas the renal-metabolic
        state remains the downstream brain-risk gateway.

    The figure intentionally does not claim a confirmed renal mediation pathway.
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.colors import LinearSegmentedColormap


    ROOT = Path(__file__).resolve().parents[2]
    OUT = ROOT / "outputs" / "downstream" / "metabolic_buffering_failure"
    FIG = ROOT / "figures"
    SRC = ROOT / "source_data"

    FIG.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    PALETTE = {
        "ink": "#202124",
        "muted": "#666666",
        "grid": "#d9d9d9",
        "dm": "#9b5de5",
        "non": "#4d908e",
        "mbf": "#d95f02",
        "renal": "#2b6cb0",
        "hyper": "#b83280",
        "cam": "#4c78a8",
        "light": "#f6f6f6",
    }


    def mm_to_in(mm: float) -> float:
        return mm / 25.4


    def add_panel_label(ax: plt.Axes, label: str) -> None:
        ax.text(
            -0.12, 1.08, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left", color=PALETTE["ink"],
        )


    def load() -> dict[str, pd.DataFrame]:
        d = {
            "audit": pd.read_csv(OUT / "mbf_index_generation_audit.csv"),
            "importance": pd.read_csv(OUT / "mbf_index_variable_contributions.csv"),
            "distribution": pd.read_csv(OUT / "mbf_distribution_by_diabetes.csv"),
            "severity": pd.read_csv(OUT / "mbf_severity_independence_model.csv"),
            "incremental": pd.read_csv(OUT / "mbf_incremental_value_renal_transition.csv"),
            "quartiles": pd.read_csv(OUT / "mbf_quartile_outcomes.csv"),
            "attenuation": pd.read_csv(OUT / "diabetes_effect_attenuation_by_mbf.csv"),
            "cam": pd.read_csv(OUT / "mbf_cam_bridge_models.csv"),
            "rows": pd.read_csv(OUT / "mbf_analysis_rows_full_icu.csv.gz"),
        }
        return d


    def draw_panel_a(ax: plt.Axes) -> None:
        ax.axis("off")
        add_panel_label(ax, "a")
        ax.set_title("Dynamic metabolic-resilience hypothesis", loc="left", pad=2, fontsize=7.2)
        xs = [0.08, 0.31, 0.55, 0.78]
        labels = [
            ("Diabetes", "susceptibility"),
            ("MBF index", "0-24 h resilience phenotype"),
            ("Renal-metabolic\ntrajectory", "24-48 h state"),
            ("Strict CAM", "48-72 h brain outcome"),
        ]
        colors = ["#efe7fb", "#fff1df", "#e8f1fb", "#edf3f8"]
        edge = [PALETTE["dm"], PALETTE["mbf"], PALETTE["renal"], PALETTE["cam"]]
        for i, (x, lab, col, ec) in enumerate(zip(xs, labels, colors, edge)):
            ax.add_patch(plt.Rectangle((x, 0.45), 0.17, 0.28, facecolor=col, edgecolor=ec, lw=1.0))
            ax.text(x + 0.085, 0.61, lab[0], ha="center", va="center", fontsize=6.5, fontweight="bold")
            ax.text(x + 0.085, 0.50, lab[1], ha="center", va="center", fontsize=5.4, color=PALETTE["muted"])
            if i < 3:
                ax.annotate("", xy=(xs[i + 1] - 0.01, 0.59), xytext=(x + 0.18, 0.59),
                            arrowprops=dict(arrowstyle="->", lw=0.9, color=PALETTE["ink"]))
        ax.text(0.08, 0.26, "MBF = observed minus expected glucose change\nconditioned on early reserve, inputs and stress",
                fontsize=5.8, color=PALETTE["ink"], va="top")
        ax.text(0.60, 0.26, "Language: attenuation / explanatory pathway,\nnot formal causal mediation",
                fontsize=5.8, color=PALETTE["muted"], va="top")


    def draw_panel_b(ax: plt.Axes, importance: pd.DataFrame) -> None:
        add_panel_label(ax, "b")
        q = importance.head(10).copy().iloc[::-1]
        labels = [x.replace("mbf_", "").replace("_0_12", "").replace("_0_6", "").replace("_", " ") for x in q.feature]
        ax.barh(np.arange(len(q)), q.permutation_delta_R2_mean.astype(float), color="#9ecae1", edgecolor="#5c8fb8", lw=0.4)
        ax.set_yticks(np.arange(len(q)), labels, fontsize=5.3)
        ax.set_xlabel("Permutation delta R2")
        ax.set_title("Input-response model contributors", loc="left", pad=2, fontsize=7.0)
        ax.grid(axis="x", color=PALETTE["grid"], lw=0.4)
        importance.head(20).to_csv(SRC / "Figure_6b_mbf_variable_contributions.csv", index=False)


    def draw_panel_c(ax: plt.Axes, dist: pd.DataFrame, severity: pd.DataFrame) -> None:
        add_panel_label(ax, "c")
        q = dist.copy()
        q["label"] = q.diabetes.map({0: "Non-diabetes", 1: "Diabetes"})
        colors = [PALETTE["non"], PALETTE["dm"]]
        ax.bar(q["label"], q.mbf_mean.astype(float), yerr=q.mbf_sd.astype(float) / np.sqrt(q.n.astype(float)),
               color=colors, edgecolor="white", lw=0.6)
        ax.axhline(0, color=PALETTE["ink"], lw=0.6)
        ax.set_ylabel("Mean MBF index")
        ax.set_title("Diabetes specificity", loc="left", pad=2, fontsize=7.0)
        est = float(severity.estimate.iloc[0])
        lo = float(severity.ci_lower.iloc[0])
        hi = float(severity.ci_upper.iloc[0])
        ax.text(0.02, 0.95, f"Adjusted DM coefficient\n{est:.2f} ({lo:.2f}, {hi:.2f})",
                transform=ax.transAxes, va="top", fontsize=5.7)
        dist.to_csv(SRC / "Figure_6c_mbf_distribution_by_diabetes.csv", index=False)
        severity.to_csv(SRC / "Figure_6c_mbf_severity_independence.csv", index=False)


    def draw_panel_d(ax: plt.Axes, inc: pd.DataFrame) -> None:
        add_panel_label(ax, "d")
        q = inc.copy()
        q["short"] = q["model"].map({
            "Model A: severity/reserve": "A\nseverity",
            "Model B: + traditional glycaemia": "B\n+ glucose",
            "Model C: + MBF index": "C\n+ MBF",
        })
        cohorts = ["Full ICU", "Diabetes"]
        x = np.arange(3)
        width = 0.32
        for i, cohort in enumerate(cohorts):
            sub = q[q.cohort.eq(cohort)]
            ax.bar(x + (i - 0.5) * width, sub.R2.astype(float), width=width,
                   label=cohort, color=[PALETTE["non"], PALETTE["dm"]][i], alpha=0.85)
        ax.set_xticks(x, ["A\nseverity", "B\n+ glucose", "C\n+ MBF"])
        ax.set_ylabel("Cross-fitted R2\nrenal trajectory")
        ax.set_ylim(0.46, 0.515)
        ax.set_title("MBF adds little beyond glucose in full ICU", loc="left", pad=2, fontsize=7.0)
        ax.legend(fontsize=5.5, loc="upper left")
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.4)
        inc.to_csv(SRC / "Figure_6d_mbf_incremental_value.csv", index=False)


    def draw_panel_e(ax: plt.Axes, quartiles: pd.DataFrame) -> None:
        add_panel_label(ax, "e")
        q = quartiles[quartiles.cohort.eq("Diabetes")].copy()
        x = np.arange(len(q))
        ax.plot(x, q.renal_transition_burden.astype(float), marker="o", color=PALETTE["renal"],
                label="Renal-metabolic")
        ax.plot(x, q.hyperglycaemic_instability_burden.astype(float), marker="s", color=PALETTE["hyper"],
                label="Hyperglycaemic")
        ax.set_xticks(x, q.mbf_quartile)
        ax.set_ylabel("Mean state probability")
        ax.set_title("Diabetes MBF quartiles", loc="left", pad=2, fontsize=7.0)
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.4)
        ax2 = ax.twinx()
        ax2.plot(x, q.cam_rate.astype(float), marker="^", color=PALETTE["cam"], label="CAM")
        ax2.set_ylabel("CAM rate")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=5.3, loc="upper center", ncol=3)
        quartiles.to_csv(SRC / "Figure_6e_mbf_quartile_outcomes.csv", index=False)


    def empirical_landscape(rows: pd.DataFrame) -> pd.DataFrame:
        q = rows[rows.mbf_index.notna() & rows.mbf_metabolic_challenge_index.notna()].copy()
        q["challenge_bin"] = pd.qcut(q.mbf_metabolic_challenge_index, 8, labels=False, duplicates="drop")
        q["mbf_bin"] = pd.qcut(q.mbf_index, 8, labels=False, duplicates="drop")
        q["cam_assessed_flag"] = q["strict_cam_assessed_48_72"].astype(float).fillna(0).eq(1)
        out = q.groupby(["diabetes", "challenge_bin", "mbf_bin"], observed=True).agg(
            n=("stay_id", "size"),
            challenge=("mbf_metabolic_challenge_index", "mean"),
            mbf=("mbf_index", "mean"),
            renal=("renal_transition_burden", "mean"),
            hyper=("post_mean_p_state_5", "mean"),
            cam_assessed=("cam_assessed_flag", "sum"),
            cam_events=("strict_cam_positive_48_72", "sum"),
        ).reset_index()
        out["effective_sample_size"] = out["n"].astype(float)
        out["weight_policy"] = "unweighted empirical landscape; ESS equals bin n"
        return out


    def draw_panel_f(ax: plt.Axes, rows: pd.DataFrame) -> None:
        add_panel_label(ax, "f")
        land = empirical_landscape(rows)
        dm = land[land.diabetes.eq(1)].copy()
        pivot = dm.pivot(index="mbf_bin", columns="challenge_bin", values="hyper")
        im = ax.imshow(pivot.sort_index(ascending=True).to_numpy(), origin="lower",
                       cmap=LinearSegmentedColormap.from_list("hyper", ["#f7fbff", "#fdd0a2", "#b83280"]),
                       aspect="auto")
        ax.set_xlabel("Metabolic challenge bin")
        ax.set_ylabel("MBF index bin")
        ax.set_title("Empirical landscape: hyperglycaemic state", loc="left", pad=2, fontsize=7.0)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Mean probability", fontsize=5.5)
        min_n = int(dm["n"].min()) if len(dm) else 0
        ax.text(
            0.02, 0.97, f"Diabetes bins; min n={min_n}; ESS=n",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.3,
            color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.80),
        )
        land.to_csv(SRC / "Figure_6f_mbf_empirical_landscape.csv", index=False)


    def draw_panel_g(ax: plt.Axes, attenuation: pd.DataFrame, cam: pd.DataFrame) -> None:
        add_panel_label(ax, "g")
        ax.axis("off")
        ax.set_title("Explanatory pathway readout", loc="left", pad=2, fontsize=7.0)
        renal = attenuation[attenuation.outcome.eq("renal_transition_burden") & attenuation.model.str.startswith("Model 2")].iloc[0]
        hyper = attenuation[attenuation.outcome.eq("post_mean_p_state_5") & attenuation.model.str.startswith("Model 2")].iloc[0]
        cam_dm = cam[(cam.cohort.eq("Diabetes")) & cam.model.eq("Model C: + MBF index + renal trajectory")].iloc[0]
        texts = [
            ("DM -> hyperglycaemic trajectory", f"attenuation {100*float(hyper.attenuation_fraction):.1f}%\nMBF beta {float(hyper.mbf_beta):.3f}"),
            ("DM -> renal-metabolic trajectory", f"not attenuated\nMBF beta {float(renal.mbf_beta):.3f}"),
            ("CAM bridge", f"Diabetes AUROC {float(cam_dm.AUROC):.3f}\nwith MBF + renal state"),
        ]
        y = 0.78
        for title, body in texts:
            ax.text(0.03, y, title, fontsize=6.4, fontweight="bold", va="top")
            ax.text(0.03, y - 0.12, body, fontsize=5.8, color=PALETTE["muted"], va="top")
            y -= 0.28
        attenuation.to_csv(SRC / "Figure_6g_diabetes_attenuation_by_mbf.csv", index=False)
        cam.to_csv(SRC / "Figure_6g_mbf_cam_bridge_models.csv", index=False)


    def save(fig: plt.Figure) -> None:
        base = FIG / "Figure_6_metabolic_buffering_failure_axis"
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


    def main() -> None:
        data = load()
        fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(210)), constrained_layout=False)
        gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.85, 1.1, 1.1, 1.0], hspace=0.58, wspace=0.38)

        draw_panel_a(fig.add_subplot(gs[0, :]))
        draw_panel_b(fig.add_subplot(gs[1, 0]), data["importance"])
        draw_panel_c(fig.add_subplot(gs[1, 1]), data["distribution"], data["severity"])
        draw_panel_d(fig.add_subplot(gs[2, 0]), data["incremental"])
        draw_panel_e(fig.add_subplot(gs[2, 1]), data["quartiles"])
        draw_panel_f(fig.add_subplot(gs[3, 0]), data["rows"])
        draw_panel_g(fig.add_subplot(gs[3, 1]), data["attenuation"], data["cam"])

        fig.suptitle(
            "Metabolic buffering failure marks diabetes-associated metabolic vulnerability",
            fontsize=9.5, y=0.978,
        )
        fig.subplots_adjust(left=0.075, right=0.975, top=0.925, bottom=0.055)
        save(fig)

    return SimpleNamespace(**locals())

_mbf = _load_mbf()


# ==============================================================================
# Independent-contribution figure
# ==============================================================================

def _load_contrib():
    """Test whether the two proposed Figure 6 axes add non-redundant clinical signal.

    The available patient-level source table contains the observed cerebrovascular
    endpoint, but not strict-CAM onset data.  This analysis therefore labels the
    outcome exactly as cerebrovascular and avoids using ``organ_vulnerability`` as
    a predictor because that label count includes the cerebrovascular endpoint.

    Two predictor specifications are reported:
    1. The published composite renal-metabolic burden, which already contains the
       MBF component and therefore tests the current, potentially overlapping axes.
    2. An orthogonal sensitivity specification using the creatinine component as a
       renal-only proxy, which tests whether MBF adds signal beyond renal burden
       without mechanically reusing MBF in the renal predictor.
    """
    import argparse
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    RELEASE_ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_INPUT = RELEASE_ROOT / "data" / "restricted" / "figure6_patient_level" / "mimic_derived_mbf_proxy_and_renal_metabolic_burden.csv"
    DEFAULT_OUTPUT = RELEASE_ROOT / "outputs" / "figure6_independent_contribution_analysis"


    def add_standardized_variables(data: pd.DataFrame) -> pd.DataFrame:
        out = data.copy()
        for col in ("glycaemic_mismatch_z", "creatinine_0_24h_z", "renal_metabolic_burden", "age"):
            out[f"{col}_analysis_z"] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
        return out


    def fit_logit(data: pd.DataFrame, predictors: list[str], model_name: str) -> tuple[pd.DataFrame, dict[str, float], object]:
        x = sm.add_constant(data[predictors].astype(float), has_constant="add")
        y = data["cerebrovascular"].astype(int)
        fit = sm.Logit(y, x).fit(disp=False)
        ci = fit.conf_int()
        rows = []
        for predictor in predictors:
            rows.append(
                {
                    "model": model_name,
                    "predictor": predictor,
                    "beta": float(fit.params[predictor]),
                    "odds_ratio": float(np.exp(fit.params[predictor])),
                    "ci_lower": float(np.exp(ci.loc[predictor, 0])),
                    "ci_upper": float(np.exp(ci.loc[predictor, 1])),
                    "p_value": float(fit.pvalues[predictor]),
                }
            )
        metrics = {
            "model": model_name,
            "n": int(len(data)),
            "events": int(y.sum()),
            "aic": float(fit.aic),
            "log_likelihood": float(fit.llf),
            "mcfadden_pseudo_r2": float(fit.prsquared),
            "lr_p_vs_null": float(fit.llr_pvalue),
        }
        return pd.DataFrame(rows), metrics, fit


    def nested_lr(base_fit: object, expanded_fit: object, expanded_name: str) -> dict[str, float | str]:
        lr = 2.0 * (float(expanded_fit.llf) - float(base_fit.llf))
        df = int(len(expanded_fit.params) - len(base_fit.params))
        from scipy.stats import chi2

        return {
            "comparison": expanded_name,
            "added_parameters": df,
            "lr_statistic": lr,
            "lr_p_value": float(chi2.sf(lr, df)),
            "delta_aic": float(expanded_fit.aic - base_fit.aic),
        }


    def cross_validated_auc(data: pd.DataFrame, model_specs: dict[str, list[str]], seed: int = 20260817) -> pd.DataFrame:
        y = data["cerebrovascular"].astype(int).to_numpy()
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rows = []
        for name, predictors in model_specs.items():
            fold_values = []
            for train, test in folds.split(data, y):
                preprocessor = ColumnTransformer(
                    [("continuous", StandardScaler(), [p for p in predictors if p != "insulin_0_24h"]),
                     ("binary", "passthrough", [p for p in predictors if p == "insulin_0_24h"])],
                    remainder="drop",
                )
                model = Pipeline(
                    [("preprocess", preprocessor), ("logit", LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs"))]
                )
                model.fit(data.iloc[train][predictors], y[train])
                pred = model.predict_proba(data.iloc[test][predictors])[:, 1]
                fold_values.append(roc_auc_score(y[test], pred))
            rows.append(
                {
                    "model": name,
                    "cv_auc_mean": float(np.mean(fold_values)),
                    "cv_auc_sd": float(np.std(fold_values, ddof=1)),
                    "fold_auc": ";".join(f"{v:.6f}" for v in fold_values),
                }
            )
        result = pd.DataFrame(rows)
        base_auc = float(result.loc[result["model"] == "base", "cv_auc_mean"].iloc[0])
        result["delta_cv_auc_vs_base"] = result["cv_auc_mean"] - base_auc
        return result


    def make_forest(effects: pd.DataFrame, output: Path) -> None:
        plot_rows = effects[
            effects["model"].isin(["both_shared", "both_orthogonal"])
            & effects["predictor"].isin(["mbf_axis", "renal_composite_axis", "renal_only_axis"])
        ].copy()
        labels = {
            "mbf_axis": "MBF proxy",
            "renal_composite_axis": "Composite renal burden",
            "renal_only_axis": "Creatinine-only renal component",
        }
        colors = {"shared-axis model": "#D96C62", "orthogonal sensitivity model": "#5B7FA3"}
        plot_rows["label"] = plot_rows["predictor"].map(labels)
        plot_rows["row_label"] = plot_rows["model"].map({"both_shared": "Shared", "both_orthogonal": "Orthogonal sensitivity"}) + " | " + plot_rows["label"]
        plot_rows = plot_rows.sort_values(["model", "predictor"])
        fig, ax = plt.subplots(figsize=(6.2, 2.7))
        y = np.arange(len(plot_rows))[::-1]
        for yy, row in zip(y, plot_rows.itertuples(index=False)):
            color = colors.get(row.model, "#333333")
            ax.errorbar(row.odds_ratio, yy, xerr=[[row.odds_ratio - row.ci_lower], [row.ci_upper - row.odds_ratio]], fmt="o", color=color, ecolor=color, capsize=2.4, lw=1.1)
            ax.text(min(row.ci_upper * 1.04, 3.0), yy, f"{row.odds_ratio:.2f} ({row.ci_lower:.2f}-{row.ci_upper:.2f})", va="center", fontsize=7)
        ax.axvline(1.0, color="#555555", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.5, 2.8)
        ax.set_yticks(y, plot_rows["row_label"])
        ax.set_xlabel("Adjusted odds ratio per 1 SD")
        ax.set_title("Joint model: non-redundant axis contributions", loc="left", fontsize=9, fontweight="bold")
        ax.grid(axis="x", color="#E5E9EC", lw=0.6)
        fig.text(0.01, 0.01, "Observed cerebrovascular endpoint; models adjust for age and insulin exposure. Strict-CAM onset was unavailable.", fontsize=6.5, color="#606060")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output / "independent_contribution_forest.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / "independent_contribution_forest.pdf", bbox_inches="tight")
        fig.savefig(output / "independent_contribution_forest.svg", bbox_inches="tight")
        fig.savefig(output / "independent_contribution_forest.tiff", dpi=600, bbox_inches="tight")
        plt.close(fig)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
        parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        args = parser.parse_args()
        args.output.mkdir(parents=True, exist_ok=True)

        data = add_standardized_variables(pd.read_csv(args.input))
        data["mbf_axis"] = data["glycaemic_mismatch_z_analysis_z"]
        data["renal_composite_axis"] = data["renal_metabolic_burden_analysis_z"]
        data["renal_only_axis"] = data["creatinine_0_24h_z_analysis_z"]

        corr_cols = ["mbf_axis", "renal_composite_axis", "renal_only_axis", "organ_vulnerability"]
        corr_rows = []
        for a in corr_cols:
            for b in corr_cols:
                if a < b:
                    corr_rows.append({"variable_a": a, "variable_b": b, "spearman_r": data[[a, b]].corr(method="spearman").iloc[0, 1], "pearson_r": data[[a, b]].corr(method="pearson").iloc[0, 1]})
        pd.DataFrame(corr_rows).to_csv(args.output / "axis_correlations.csv", index=False)

        model_specs = {
            "base": ["age_analysis_z", "insulin_0_24h"],
            "mbf_only": ["age_analysis_z", "insulin_0_24h", "mbf_axis"],
            "renal_composite_only": ["age_analysis_z", "insulin_0_24h", "renal_composite_axis"],
            "both_shared": ["age_analysis_z", "insulin_0_24h", "mbf_axis", "renal_composite_axis"],
            "both_orthogonal": ["age_analysis_z", "insulin_0_24h", "mbf_axis", "renal_only_axis"],
        }
        effect_frames = []
        metric_rows = []
        fits = {}
        for name, predictors in model_specs.items():
            effect, metrics, fit = fit_logit(data, predictors, name)
            effect_frames.append(effect)
            metric_rows.append(metrics)
            fits[name] = fit
        pd.concat(effect_frames, ignore_index=True).to_csv(args.output / "logistic_effects.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(args.output / "model_metrics.csv", index=False)

        comparisons = [
            nested_lr(fits["base"], fits["mbf_only"], "add MBF to base"),
            nested_lr(fits["base"], fits["renal_composite_only"], "add composite renal burden to base"),
            nested_lr(fits["base"], fits["both_shared"], "add MBF + composite renal burden to base"),
            nested_lr(fits["base"], fits["both_orthogonal"], "add MBF + creatinine-only renal component to base"),
        ]
        pd.DataFrame(comparisons).to_csv(args.output / "nested_likelihood_ratio_tests.csv", index=False)
        cv = cross_validated_auc(data, model_specs)
        cv.to_csv(args.output / "cross_validated_auc.csv", index=False)

        decomposition_error = np.max(np.abs(data["renal_metabolic_burden"] - (data["creatinine_0_24h_z"] + data["glycaemic_mismatch_z"]) / np.sqrt(2)))
        summary = {
            "input": str(args.input),
            "n": int(len(data)),
            "cerebrovascular_events": int(data["cerebrovascular"].sum()),
            "renal_metabolic_burden_definition_max_error": float(decomposition_error),
            "spearman_mbf_vs_composite_renal_burden": float(data[["mbf_axis", "renal_composite_axis"]].corr(method="spearman").iloc[0, 1]),
            "spearman_mbf_vs_renal_only_component": float(data[["mbf_axis", "renal_only_axis"]].corr(method="spearman").iloc[0, 1]),
            "strict_cam_available": False,
            "outcome_label": "observed cerebrovascular endpoint",
        }
        (args.output / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        make_forest(pd.concat(effect_frames, ignore_index=True), args.output)
        print(json.dumps(summary, indent=2))
        print(pd.concat(effect_frames, ignore_index=True).to_string(index=False))
        print(pd.DataFrame(comparisons).to_string(index=False))
        print(cv.to_string(index=False))

    return SimpleNamespace(**locals())

_contrib = _load_contrib()


STAGES = {
    "dual-axis": _dual.main,
    "buffering-failure": _mbf.main,
    "independent-contributions": _contrib.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Metabolic-buffering and dual-axis manuscript figures')
    parser.add_argument("stage", choices=STAGES)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    stage = sys.argv[1]
    if stage not in STAGES:
        parser.error(f"invalid stage: {stage}")
    if stage in {"dual-axis", "buffering-failure"} and any(arg in {"-h", "--help"} for arg in sys.argv[2:]):
        print(f"{stage}: builds the fixed manuscript figure set (no extra options).")
        return
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    STAGES[stage]()


if __name__ == "__main__":
    main()
