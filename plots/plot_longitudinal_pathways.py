"""Create the longitudinal state-transition figure and its causal DAG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[2]
DIAB = PACKAGE / "outputs" / "downstream" / "longitudinal_pathway"
FULL = PACKAGE / "outputs" / "downstream" / "longitudinal_pathway_full_icu"
ARCH = PACKAGE / "outputs" / "downstream" / "longitudinal_pathway_archived_state_sensitivity"
FIG = PACKAGE / "figures"
SRC = PACKAGE / "source_data"

STATE_NAMES = [
    "Inflammatory–\nhaemodynamic",
    "Mixed\nvulnerability 1",
    "Stable\nmetabolic",
    "Renal–metabolic\nvulnerability",
    "Mixed\nvulnerability 2",
    "Hyperglycaemic\ninstability",
]
SHORT = ["Inflamm.–haem.", "Mixed 1", "Stable", "Renal–metab.", "Mixed 2", "Hyperglyc."]
COLORS = {
    "navy": "#254B73", "blue": "#4C78A8", "teal": "#3B8C88",
    "orange": "#D88932", "red": "#B44E4A", "purple": "#7A6AA6",
    "grey": "#777777", "light": "#E9EDF2", "dark": "#252525",
    "green": "#4D8B5A",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
        "axes.linewidth": 0.75, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.3,
    })


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")


def load_and_export_source_data() -> dict[str, pd.DataFrame]:
    SRC.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(FULL / "standardized_treatment_transition_atlas.csv")
    ref = atlas[atlas.regime.eq("stable_energy_no_steroid")][
        ["current_state_id", "next_state_id", "current_state", "next_state",
         "standardized_probability", "n_stays", "subjects"]
    ].rename(columns={"standardized_probability": "reference_probability"})
    adv = atlas[atlas.regime.eq("unstable_energy_prescribed_steroid")][
        ["current_state_id", "next_state_id", "standardized_probability"]
    ].rename(columns={"standardized_probability": "adverse_probability"})
    atlas_contrast = ref.merge(adv, on=["current_state_id", "next_state_id"])
    atlas_contrast["probability_difference"] = (
        atlas_contrast.adverse_probability - atlas_contrast.reference_probability
    )
    atlas_contrast.to_csv(SRC / "Figure_5b_standardized_transition_atlas.csv", index=False)

    curves = []
    for cohort, root in [("Diabetes PPD-overlap", DIAB), ("Full ICU secondary", FULL)]:
        q = pd.read_csv(root / "continuous_state_outcome_curve.csv")
        q["cohort"] = cohort
        curves.append(q)
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(SRC / "Figure_5c_continuous_state_outcome_curve.csv", index=False)

    support_parts = []
    for cohort, path, column in [
        ("Primary pathway cohort: diabetes", DIAB / "longitudinal_state_transition_analysis_rows_primary_0_48.csv.gz", "primary_renal_transition_burden"),
        ("Secondary outcome-anchor cohort: all ICU", FULL / "longitudinal_state_transition_analysis_rows_full_icu.csv.gz", "primary_renal_transition_burden"),
    ]:
        values = pd.read_csv(path, usecols=[column])[column].dropna().to_numpy(float)
        for q in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
            support_parts.append({"cohort": cohort, "quantile": q,
                                  "renal_transition_burden": float(np.quantile(values, q)),
                                  "n": len(values)})
    support = pd.DataFrame(support_parts)
    support.to_csv(SRC / "Figure_5c_empirical_support.csv", index=False)

    regime = pd.read_csv(DIAB / "joint_regime_standardized_effects.csv")
    regime = regime[regime.analysis_arm.eq("all")].copy()
    regime.to_csv(SRC / "Figure_5d_joint_regime_effects.csv", index=False)

    pathway_parts = []
    for label, root in [
        ("Primary diabetes state model", DIAB),
        ("Full ICU secondary", FULL),
        ("Archived-state sensitivity", ARCH),
    ]:
        q = pd.read_csv(root / "longitudinal_interventional_pathway_effects.csv")
        q = q[q.analysis_arm.eq("all")].copy(); q["analysis"] = label
        pathway_parts.append(q)
    for label, root in [
        ("Primary + landmark IPCW", DIAB),
        ("Full ICU + landmark IPCW", FULL),
    ]:
        q = pd.read_csv(root / "landmark_selection_weighted_pathway_effects.csv")
        q["analysis"] = label; q["analysis_arm"] = "selection_weighted"
        pathway_parts.append(q)
    pathway = pd.concat(pathway_parts, ignore_index=True, sort=False)
    pathway.to_csv(SRC / "Figure_5e_longitudinal_pathway_sensitivity.csv", index=False)

    fals_parts = []
    for cohort, root in [("Diabetes PPD-overlap", DIAB), ("Full ICU secondary", FULL)]:
        reg = pd.read_csv(root / "joint_regime_standardized_effects.csv")
        reg = reg[(reg.analysis_arm.eq("all")) & reg.estimand.eq("renal_transition_burden")]
        wide = reg.set_index("regime").estimate
        fals_parts.append({
            "cohort": cohort, "exposure": "Energy instability + prescribed steroid",
            "estimate": float(wide["unstable_energy_prescribed_steroid"] - wide["stable_energy_no_steroid"]),
            "type": "main joint contrast",
        })
        f = pd.read_csv(root / "falsification_exposure_effects.csv")
        f = f[f.estimand.eq("renal_transition_burden_difference_1_vs_0")]
        for r in f.itertuples():
            fals_parts.append({"cohort": cohort, "exposure": r.exposure,
                               "estimate": r.estimate, "ci_lower": r.ci_lower,
                               "ci_upper": r.ci_upper, "type": "falsification exposure"})
    fals = pd.DataFrame(fals_parts)
    fals.to_csv(SRC / "Figure_5f_falsification_effects.csv", index=False)
    bridge = pd.read_csv(DIAB / "ppd_clinical_continuous_bridge.csv")
    bridge.to_csv(SRC / "Figure_5f_ppd_clinical_bridge.csv", index=False)
    return {"atlas": atlas_contrast, "curve": curve, "support": support, "regime": regime,
            "pathway": pathway, "fals": fals, "bridge": bridge}


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.035, y=0.91)
    ax.text(0.00, 0.90, "Discovery layer (hypothesis-generating)", fontweight="bold", fontsize=7.5)
    y = 0.60
    boxes = [
        (0.00, "PPD-EHR\nrepresentations", COLORS["navy"]),
        (0.17, "Diabetes-associated\nlatent trajectories", COLORS["blue"]),
        (0.35, "Candidate metabolic\nrisk regions", COLORS["teal"]),
        (0.52, "Independent clinical\nstate resolution", COLORS["purple"]),
    ]
    for x, text, color in boxes:
        r = patches.FancyBboxPatch((x, y), 0.145, 0.23, boxstyle="round,pad=0.012",
                                   fc=color, ec="none", alpha=0.95)
        ax.add_patch(r); ax.text(x + 0.0725, y + 0.115, text, color="white", ha="center", va="center", fontsize=6.3)
    for x0, x1 in [(0.145, 0.17), (0.315, 0.35)]:
        ax.annotate("", (x1, y + .115), (x0, y + .115), arrowprops=dict(arrowstyle="->", lw=1, color=COLORS["grey"]))
    ax.annotate("", (0.52, y + .115), (0.495, y + .115),
                arrowprops=dict(arrowstyle="->", lw=1, ls="--", color=COLORS["grey"]))
    ax.text(0.70, 0.90, "Longitudinal analysis timeline", fontweight="bold", fontsize=7.5)
    x0, widths = 0.70, [0.060, 0.060, 0.070, 0.060]
    labels = ["Baseline\n0–6 h", "Exposure\n6–24 h", "State trajectory\n24–48 h", "Strict CAM\n48–72 h"]
    colors = [COLORS["grey"], COLORS["orange"], COLORS["purple"], COLORS["red"]]
    xpos = x0
    for i, (w, lab, col) in enumerate(zip(widths, labels, colors)):
        r = patches.FancyBboxPatch((xpos, y), w, .23, boxstyle="round,pad=.008", fc=col, ec="none", alpha=.92)
        ax.add_patch(r); ax.text(xpos + w/2, y + .115, lab, ha="center", va="center", color="white", fontsize=5.8)
        if i < len(widths)-1:
            ax.annotate("", (xpos+w+.008, y+.115), (xpos+w, y+.115), arrowprops=dict(arrowstyle="->", lw=.9, color=COLORS["dark"]))
        xpos += w + .014
    ax.text(0.00, 0.20, "PPD and clinical hard-state taxonomies are non-equivalent.\nThe clinical GMM is fitted in a patient-disjoint, outcome-free 0–48 h arm.",
            fontsize=5.9, color=COLORS["dark"], va="top")
    ax.text(0.70, 0.20, "Causal DAG: Supplementary Figure.\nMissing CAM, death and discharge are explicit.",
            fontsize=5.9, color=COLORS["dark"], va="top")


def draw_panel_b(ax: plt.Axes, atlas: pd.DataFrame) -> None:
    panel_label(ax, "b")
    mat = atlas.pivot(index="current_state_id", columns="next_state_id", values="probability_difference").reindex(index=range(6), columns=range(6)).to_numpy() * 100
    lim = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.5)
    # Use vector cells so SVG/PDF exports do not contain a low-resolution
    # embedded heatmap raster.
    x_edges = np.arange(mat.shape[1] + 1) - 0.5
    y_edges = np.arange(mat.shape[0] + 1) - 0.5
    im = ax.pcolormesh(
        x_edges, y_edges, mat,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)
    for i in range(6):
        for j in range(6):
            val = mat[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(val) > lim * .55 else COLORS["dark"])
    ax.set_xticks(range(6), SHORT, rotation=40, ha="right")
    ax.set_yticks(range(6), SHORT)
    ax.set_xlabel("Next state (24–30 h)")
    ax.set_ylabel("Current state (18–24 h)")
    ax.set_title("Standardized transition contrast:\nunstable energy + steroid vs reference", loc="left", pad=4)
    # Draw a discrete vector color scale; Matplotlib's default SVG colorbar
    # embeds a low-resolution raster gradient.
    cb = ax.inset_axes([1.02, 0.0, 0.035, 1.0])
    cmap = mpl.colormaps["RdBu_r"]
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    n_steps = 40
    for i in range(n_steps):
        v0 = -lim + (2 * lim) * i / n_steps
        v1 = -lim + (2 * lim) * (i + 1) / n_steps
        vm = (v0 + v1) / 2
        cb.add_patch(patches.Rectangle((0, i / n_steps), 1, 1 / n_steps,
                                       facecolor=cmap(norm(vm)), edgecolor="none"))
    cb.set_xlim(0, 1); cb.set_ylim(0, 1); cb.set_xticks([])
    cb.yaxis.tick_right()
    cb.yaxis.set_label_position("right")
    cb.set_yticks([0, .5, 1], [f"{-lim:.0f}", "0", f"{lim:.0f}"])
    cb.tick_params(labelsize=5.5, length=2)
    cb.set_ylabel("Probability difference (percentage points)", fontsize=6, rotation=270, labelpad=12)


def draw_panel_c(ax: plt.Axes, curve: pd.DataFrame, support: pd.DataFrame) -> None:
    panel_label(ax, "c")
    colors = {"Diabetes PPD-overlap": COLORS["red"], "Full ICU secondary": COLORS["navy"]}
    for cohort, q in curve[curve.renal_transition_burden.notna()].groupby("cohort"):
        q = q.sort_values("renal_transition_burden")
        support_max = 0.45 if cohort == "Diabetes PPD-overlap" else 0.75
        q = q[q.renal_transition_burden.le(support_max)]
        x = q.renal_transition_burden.to_numpy(float)
        y = q.standardized_cam_risk.to_numpy(float)
        lo = q.ci_lower.to_numpy(float); hi = q.ci_upper.to_numpy(float)
        ls = "--" if cohort == "Diabetes PPD-overlap" else "-"
        ax.plot(x, y, color=colors[cohort], label=cohort, linestyle=ls)
        ax.fill_between(x, lo, hi, color=colors[cohort], alpha=.16, lw=0)
    ax.set_xlabel("Mean renal–metabolic state probability, 24–48 h")
    ax.set_ylabel("Standardized strict-CAM risk")
    ax.set_title("Continuous state trajectory and\nsubsequent acute brain dysfunction", loc="left", pad=4)
    ax.legend(loc="upper left", fontsize=5.8)
    ax.set_xlim(0, .75); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    for cohort, y0, y1, color in [
        ("Primary pathway cohort: diabetes", .002, .005, COLORS["red"]),
        ("Secondary outcome-anchor cohort: all ICU", .006, .009, COLORS["navy"]),
    ]:
        xq = support[support.cohort.eq(cohort)].renal_transition_burden.to_numpy(float)
        ax.vlines(xq, y0, y1, color=color, lw=.65, alpha=.65)
    summary = curve[(curve.cohort.eq("Full ICU secondary")) & curve.estimand.fillna("").str.contains("0.10")]
    full_rd = float(summary.standardized_cam_risk.iloc[0]) if len(summary) else np.nan
    ax.text(.98, .04,
            f"Secondary all-ICU: {1000*full_rd:.1f} per 1,000 per 0.10\n"
            "n=21,965; CAM events=1,264; support shown as rugs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=COLORS["dark"])


def draw_panel_d(ax: plt.Axes, regime: pd.DataFrame) -> None:
    panel_label(ax, "d")
    sub = ax.get_subplotspec().subgridspec(1, 2, wspace=.52)
    ax.remove()
    labels = ["Stable, no steroid", "Unstable, no steroid", "Stable, steroid", "Unstable, steroid"]
    order = ["stable_energy_no_steroid", "unstable_energy_no_steroid",
             "stable_energy_prescribed_steroid", "unstable_energy_prescribed_steroid"]
    for j, (estimand, title, color) in enumerate([
        ("renal_transition_burden", "Mean renal-state probability", COLORS["purple"]),
        ("strict_cam_risk", "Strict-CAM risk", COLORS["red"]),
    ]):
        a = plt.gcf().add_subplot(sub[0, j])
        q = regime[regime.estimand.eq(estimand)].set_index("regime").loc[order]
        y = np.arange(4)[::-1]
        a.errorbar(q.estimate, y, xerr=[q.estimate-q.ci_lower, q.ci_upper-q.estimate], fmt="o", color=color,
                   ecolor=color, capsize=2, ms=3.8)
        a.set_yticks(y, labels if j == 0 else [""]*4)
        a.set_xlabel("Probability")
        a.set_title(title, loc="left", pad=4)
        a.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=1))
        a.spines["left"].set_visible(j == 0)
        if j == 0:
            panel_label(a, "d")
    return


def draw_panel_e(ax: plt.Axes, pathway: pd.DataFrame) -> None:
    panel_label(ax, "e")
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.05, 1], wspace=.50)
    ax.remove()
    order = ["Primary diabetes state model", "Primary + landmark IPCW",
             "Full ICU secondary", "Full ICU + landmark IPCW", "Archived-state sensitivity"]
    labels = ["Primary diabetes", "Primary + selection IPCW", "Full ICU", "Full ICU + selection IPCW", "Archived state"]
    colors = {"total_effect": COLORS["navy"], "direct_effect": COLORS["orange"], "indirect_effect": COLORS["purple"]}
    left = plt.gcf().add_subplot(sub[0, 0]); right = plt.gcf().add_subplot(sub[0, 1])
    panel_label(left, "e")
    ybase = np.arange(len(order))[::-1]
    for est, offset in [("total_effect", .11), ("direct_effect", -.11)]:
        q = pathway[pathway.estimand.eq(est)].set_index("analysis").reindex(order)
        val = q.risk_difference_per_1000.to_numpy(float)
        lo = q.ci_lower_per_1000.to_numpy(float); hi = q.ci_upper_per_1000.to_numpy(float)
        left.errorbar(val, ybase+offset, xerr=[val-lo, hi-val], fmt="o", ms=3, capsize=1.7,
                      color=colors[est], label=est.replace("_", " "))
    left.axvline(0, color=COLORS["grey"], lw=.8, ls="--")
    left.set_yticks(ybase, labels); left.set_xlabel("CAM risk difference per 1,000")
    left.set_title("Total/direct RDs\n(observational)", loc="left", pad=4); left.legend(fontsize=5.5)
    q = pathway[pathway.estimand.eq("indirect_effect")].set_index("analysis").reindex(order)
    val = q.risk_difference_per_1000.to_numpy(float)
    lo = q.ci_lower_per_1000.to_numpy(float); hi = q.ci_upper_per_1000.to_numpy(float)
    right.errorbar(val, ybase, xerr=[val-lo, hi-val], fmt="o", ms=3.6, capsize=1.7, color=COLORS["purple"])
    right.axvline(0, color=COLORS["grey"], lw=.8, ls="--")
    right.set_yticks(ybase, [""]*len(order)); right.set_xlabel("Indirect risk difference per 1,000")
    right.set_title("State-pathway\nindirect RD", loc="left", pad=4)


def draw_panel_f(ax: plt.Axes, fals: pd.DataFrame, bridge: pd.DataFrame) -> None:
    panel_label(ax, "f")
    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.2, .8], wspace=.50)
    ax.remove()
    a = plt.gcf().add_subplot(sub[0, 0]); b = plt.gcf().add_subplot(sub[0, 1])
    panel_label(a, "f")
    q = fals[fals.cohort.eq("Full ICU secondary")].copy()
    display = {
        "Energy instability + prescribed steroid": "Main regime",
        "ppi_h2_any_6_24": "PPI/H2",
        "antiemetic_any_6_24": "Antiemetic",
    }
    q["label"] = q.exposure.map(display)
    q = q[q.label.notna()].sort_values("estimate")
    y = np.arange(len(q))[::-1]
    vals = q.estimate.to_numpy(float) * 1000
    lo = q.ci_lower.fillna(q.estimate).to_numpy(float) * 1000
    hi = q.ci_upper.fillna(q.estimate).to_numpy(float) * 1000
    colors = [COLORS["red"] if t == "main joint contrast" else COLORS["grey"] for t in q.type]
    for yi, v, l, u, col in zip(y, vals, lo, hi, colors):
        a.errorbar(v, yi, xerr=[[v-l], [u-v]], fmt="o", ms=3.5, capsize=2, color=col)
    a.axvline(0, color=COLORS["grey"], lw=.8, ls="--")
    a.set_yticks(y, q.label); a.set_xlabel("Change in mean renal-state probability per 1,000")
    a.set_title("Falsification exposures show non-specificity", loc="left", pad=4)
    a.text(.99, .96, "Main regime: point estimate", transform=a.transAxes,
           ha="right", va="top", fontsize=5.4, color=COLORS["dark"])

    qb = bridge.copy(); yy = np.arange(len(qb))[::-1]
    vals = qb.delta_R2.to_numpy(float)
    lo = qb.delta_R2_ci_lower.to_numpy(float); hi = qb.delta_R2_ci_upper.to_numpy(float)
    b.errorbar(vals, yy, xerr=[vals-lo, hi-vals], fmt="o", ms=3.8, capsize=2, color=COLORS["navy"])
    b.axvline(0, color=COLORS["grey"], lw=.8, ls="--")
    b.set_yticks(yy, ["Held-out test", "Subject-disjoint test"][:len(qb)])
    b.set_xlabel(r"Incremental $\Delta R^2$")
    b.set_title("Bounded PPD-EHR\ncontinuous-state information", loc="left", pad=4)
    b.text(.02, .03, "Hard-state ARI/NMI ≈ 0\n(complementary, not equivalent)", transform=b.transAxes,
           fontsize=5.6, va="bottom", color=COLORS["dark"])


def make_state_transition_figure() -> None:
    style(); FIG.mkdir(parents=True, exist_ok=True)
    data = load_and_export_source_data()
    # Nature Communications double-column width: 183 mm. Keep the physical
    # canvas fixed and do not use bbox_inches='tight', which previously
    # expanded the exported page beyond the journal width.
    fig = plt.figure(figsize=(183 / 25.4, 220 / 25.4), constrained_layout=False)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 2.1, 1.75, 1.55], width_ratios=[1, 1],
                          hspace=.62, wspace=.42, left=.125, right=.965, top=.990, bottom=.060)
    axa = fig.add_subplot(gs[0, :]); draw_panel_a(axa)
    axb = fig.add_subplot(gs[1, 0]); draw_panel_b(axb, data["atlas"])
    axc = fig.add_subplot(gs[1, 1]); draw_panel_c(axc, data["curve"], data["support"])
    axd = fig.add_subplot(gs[2, 0]); draw_panel_d(axd, data["regime"])
    axe = fig.add_subplot(gs[2, 1]); draw_panel_e(axe, data["pathway"])
    axf = fig.add_subplot(gs[3, :]); draw_panel_f(axf, data["fals"], data["bridge"])
    base = FIG / "Figure_5_longitudinal_state_transition_pathways"
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    # Matplotlib may emit an opaque RGBA TIFF; normalize to journal-friendly RGB.
    tiff_path = base.with_suffix(".tiff")
    with Image.open(tiff_path) as im:
        im.convert("RGB").save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
    print(base)


DAG_COLORS = {
    "baseline": "#777777", "context": "#6C7A89", "exposure": "#D88932",
    "state": "#7A6AA6", "outcome": "#B44E4A", "observation": "#3B8C88", "ink": "#2B2B2B",
}


def _dag_node(ax, xy, wh, text, color, *, dashed=False):
    x, y = xy
    width, height = wh
    box = patches.FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012", fc=color,
        ec=DAG_COLORS["ink"] if dashed else "none", lw=0.8,
        ls="--" if dashed else "-", alpha=0.96,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=6.5, color="white")
    return x, y, width, height


def _dag_arrow(ax, source, target, *, rad=0.0):
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    ax.add_patch(patches.FancyArrowPatch(
        (sx + sw / 2, sy + sh / 2), (tx + tw / 2, ty + th / 2),
        arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#555555",
        connectionstyle=f"arc3,rad={rad}", shrinkA=24, shrinkB=24,
    ))


def make_causal_dag() -> None:
    style()
    fig, ax = plt.subplots(figsize=(183 / 25.4, 105 / 25.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    baseline = _dag_node(ax, (0.04, 0.66), (0.18, 0.18), "Baseline severity,\nmetabolic reserve\n0–6 h", DAG_COLORS["baseline"])
    context = _dag_node(ax, (0.04, 0.24), (0.18, 0.18), "Therapeutic context\nventilation, sedation,\nvasopressors, RRT", DAG_COLORS["context"])
    exposure = _dag_node(ax, (0.30, 0.66), (0.16, 0.18), "Medication–nutrition\nexposure\n6–24 h", DAG_COLORS["exposure"])
    state = _dag_node(ax, (0.55, 0.66), (0.17, 0.18), "Renal–metabolic\nstate trajectory\n24–48 h", DAG_COLORS["state"])
    outcome = _dag_node(ax, (0.80, 0.66), (0.15, 0.18), "Underlying strict\nCAM status\n48–72 h", DAG_COLORS["outcome"])
    observation = _dag_node(ax, (0.55, 0.24), (0.17, 0.18), "CAM observation,\nsurvival and ICU\npresence", DAG_COLORS["observation"])
    observed = _dag_node(ax, (0.80, 0.24), (0.15, 0.18), "Observed strict\nCAM outcome", DAG_COLORS["outcome"], dashed=True)
    for source, target, rad in [
        (baseline, exposure, 0), (baseline, state, -0.14), (baseline, outcome, -0.23),
        (context, exposure, -0.12), (context, state, -0.05), (context, outcome, 0.10),
        (exposure, state, 0), (exposure, outcome, -0.14), (state, outcome, 0),
        (exposure, observation, 0.10), (state, observation, 0), (context, observation, 0),
        (outcome, observation, -0.20), (outcome, observed, 0), (observation, observed, 0),
    ]:
        _dag_arrow(ax, source, target, rad=rad)
    ax.text(0.04, 0.95, "Longitudinal causal-analysis DAG", fontsize=8, fontweight="bold", ha="left", va="top")
    ax.text(0.04, 0.06, "Solid arrows encode the prespecified causal/observation structure; the dashed node is observed only when CAM assessment is available.", fontsize=5.8, ha="left", va="bottom", color=DAG_COLORS["ink"])
    FIG.mkdir(parents=True, exist_ok=True)
    base = FIG / "Supplementary_Figure_longitudinal_causal_DAG"
    for extension, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 300}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(base.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)
    with Image.open(base.with_suffix(".tiff")) as image:
        image.convert("RGB").save(base.with_suffix(".tiff"), compression="tiff_lzw", dpi=(600, 600))
    print(base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("state", "dag", "all"), default="all")
    args = parser.parse_args()
    if args.figure in {"state", "all"}:
        make_state_transition_figure()
    if args.figure in {"dag", "all"}:
        make_causal_dag()


if __name__ == "__main__":
    main()
