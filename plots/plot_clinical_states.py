"""Clinical-state manuscript figures."""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace


# ==============================================================================
# Latent-trajectory figures
# ==============================================================================

def _load_latent_plot():
    import argparse
    import hashlib
    from io import BytesIO
    from pathlib import Path
    import json

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from PIL import Image


    PACKAGE = Path(__file__).resolve().parents[2]
    CLIN = PACKAGE / "data" / "derived" / "latent_trajectory"
    PPD_FIXED = PACKAGE / "outputs" / "latent_export" / "test"
    PPD = PACKAGE / "outputs" / "downstream" / "ppd"
    EICU_TRANSPORT = PACKAGE / "outputs" / "downstream" / "eicu_transport"
    OVERLAP = PACKAGE / "outputs" / "downstream" / "clinical_state"
    OUT = PACKAGE / "figures"


    def parse_args():
        p = argparse.ArgumentParser(
            description="Render Figures 3-5 using the seed-42 7:2:1 validation-discovery/test-confirmation outputs."
        )
        p.add_argument("--fixed-test-dir", type=Path, default=PPD_FIXED)
        p.add_argument("--ppd-downstream-dir", type=Path, default=PPD)
        p.add_argument("--clinical-root", type=Path, default=CLIN)
        p.add_argument("--eicu-transport-dir", type=Path, default=EICU_TRANSPORT)
        p.add_argument("--overlap-dir", type=Path, default=OVERLAP)
        p.add_argument("--output-dir", type=Path, default=OUT)
        return p.parse_args()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    COL = {
        "navy": "#244B6B", "blue": "#4F7CAC", "teal": "#4C9A9A",
        "orange": "#D98C3F", "red": "#C45A54", "grey": "#7B7B7B",
        "light": "#E9EEF2", "purple": "#8073AC", "green": "#5E9B76",
    }

    STATE_SHORT = {
        "Stable metabolic": "Stable",
        "Hyperglycaemic instability": "Hyperglycaemic\ninstability",
        "Renal-metabolic vulnerability": "Renal–metabolic\nvulnerability",
        "Inflammatory-haemodynamic stress": "Inflammatory–\nhaemodynamic",
        "Mixed vulnerability 1": "Mixed 1",
        "Mixed vulnerability 2": "Mixed 2",
    }


    def paired_replication_summary(x, y, n_boot=1000, seed=20260713):
        x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
        keep=np.isfinite(x)&np.isfinite(y); x=x[keep]; y=y[keep]
        pearson=float(np.corrcoef(x,y)[0,1]) if len(x)>1 else np.nan
        spearman=float(pd.Series(x).corr(pd.Series(y),method="spearman")) if len(x)>1 else np.nan
        sign=float((np.sign(x)==np.sign(y)).mean()) if len(x) else np.nan
        rng=np.random.default_rng(seed); boots=[]
        for _ in range(n_boot):
            idx=rng.integers(0,len(x),len(x))
            if np.std(x[idx])>0 and np.std(y[idx])>0:
                boots.append(np.corrcoef(x[idx],y[idx])[0,1])
        lo,hi=(np.quantile(boots,[.025,.975]) if boots else (np.nan,np.nan))
        return {"n":len(x),"pearson":pearson,"pearson_lo":float(lo),"pearson_hi":float(hi),"spearman":spearman,"sign":sign}


    def fixed_time_axis_label():
        audit_path=PPD_FIXED/"run_audit.json"
        if not audit_path.exists():
            return "Fixed-time landmark (h)"
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        anchor=str(audit.get("landmark_time_origin",audit.get("time_origin",""))).lower()
        if "icu" in anchor and "admission" in anchor:
            return "Hours since ICU admission"
        caveat=str(audit.get("time_origin_verification",audit.get("time_origin_caveat",""))).lower()
        if "icu-relative" in caveat or ("icu" in caveat and "admission" in caveat):
            return "Hours since ICU admission"
        if "earliest" in caveat or "bundle" in caveat:
            return "Hours since earliest retained bundle timestamp"
        return "Fixed-time landmark (h)"


    def panel(ax, label):
        ax.text(-0.15, 1.13, label, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top", clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})


    def save(fig, stem):
        fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
        tiff_path = OUT / f"{stem}.tiff"
        tiff_buffer = BytesIO()
        fig.savefig(tiff_buffer, format="png", dpi=600, bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        with Image.open(tiff_buffer) as im:
            rgb = im.convert("RGB").copy()
        tiff_buffer.close()
        tiff_path.unlink(missing_ok=True)
        rgb.save(tiff_path, compression="tiff_lzw", dpi=(600, 600))
        rgb.close()


    def fig3():
        profiles = pd.read_csv(CLIN / "latent_state_profiles.csv")
        occ = pd.read_csv(CLIN / "state_occupancy_dm_vs_nondm.csv")
        diff = pd.read_csv(CLIN / "state_transition_differences_patient_bootstrap.csv")
        fixed_diff = pd.read_csv(CLIN / "state_transition_differences_dm_minus_nondm.csv")
        outer = pd.read_csv(CLIN / "outer_state_bootstrap_summary.csv")
        eicu_diff = pd.read_csv(EICU_TRANSPORT / "eicu_state_transition_differences_dm_minus_nondm.csv")
        order = ["Stable metabolic", "Hyperglycaemic instability", "Renal-metabolic vulnerability",
                 "Inflammatory-haemodynamic stress", "Mixed vulnerability 1", "Mixed vulnerability 2"]

        fig = plt.figure(figsize=(7.2, 6.7))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[0.95, 1.15], wspace=.43, hspace=.50)
        axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

        cols = ["glucose_mean", "glucose_cv", "creatinine", "bun", "lactate", "map", "heart_rate", "spo2", "wbc"]
        hp = profiles.set_index("latent_state").reindex(order)[cols]
        hz = (hp - hp.mean()) / hp.std(ddof=0).replace(0, 1)
        sns.heatmap(hz, ax=axa, cmap=sns.diverging_palette(230, 25, as_cmap=True), center=0,
                    cbar_kws={"label": "Across-state z-score", "shrink": .7}, linewidths=.4, linecolor="white")
        axa.set_yticklabels([STATE_SHORT[x] for x in order], rotation=0)
        axa.set_xticklabels(["Glucose", "Glucose CV", "Creatinine", "BUN", "Lactate", "MAP", "Heart rate", "SpO2", "WBC"], rotation=45, ha="right")
        axa.set(xlabel="", ylabel="", title="Outcome-free clinical state profiles")
        panel(axa, "a")

        po = occ.pivot_table(index=["bin", "latent_state"], columns="diabetes", values="proportion").reset_index()
        po["difference"] = 100 * (po[1] - po[0])
        mat = po.pivot(index="latent_state", columns="bin", values="difference").reindex(order)
        lim = np.nanmax(np.abs(mat.to_numpy()))
        sns.heatmap(mat, ax=axb, cmap=sns.diverging_palette(230, 20, as_cmap=True), center=0, vmin=-lim, vmax=lim,
                    cbar_kws={"label": "DM − non-DM occupancy (percentage points)", "shrink": .75}, linewidths=.25)
        axb.set_yticklabels([STATE_SHORT[x] for x in order], rotation=0)
        axb.set_xticklabels([f"{6*(int(x)+1)}" for x in mat.columns], rotation=0)
        axb.set(xlabel="6-h window endpoint (h)", ylabel="", title="State occupancy difference over time")
        panel(axb, "b")

        keys = [
            ("Stable metabolic", "Hyperglycaemic instability", "Stable → hyper"),
            ("Renal-metabolic vulnerability", "Hyperglycaemic instability", "Renal → hyper"),
            ("Mixed vulnerability 2", "Hyperglycaemic instability", "Mixed 2 → hyper"),
            ("Hyperglycaemic instability", "Hyperglycaemic instability", "Hyper persistence"),
            ("Hyperglycaemic instability", "Mixed vulnerability 2", "Hyper → mixed 2"),
            ("Hyperglycaemic instability", "Stable metabolic", "Hyper → stable"),
        ]
        rows=[]
        for a,b,l in keys:
            r=outer[(outer.latent_state==a)&(outer.next_state==b)].iloc[0]
            rows.append((l,100*r.bootstrap_median,100*r.ci_lower,100*r.ci_upper))
        y=np.arange(len(rows))[::-1]
        vals=np.array([r[1] for r in rows]); lo=np.array([r[2] for r in rows]); hi=np.array([r[3] for r in rows])
        axc.axvline(0,color="#B8B8B8",lw=.8)
        colors=[COL["red"],COL["red"],COL["orange"],COL["grey"],COL["grey"],COL["grey"]]
        for yi,v,lw,up,c in zip(y,vals,lo,hi,colors):
            axc.errorbar(v,yi,xerr=[[v-lw],[up-v]],fmt="o",color=c,ecolor=c,capsize=2,lw=1.2)
        axc.set_yticks(y, [r[0] for r in rows]); axc.tick_params(axis="y", labelsize=5.8); axc.set_xlabel("DM − non-DM transition probability (percentage points)")
        axc.set_title("Outer state-model bootstrap")
        refits=int(pd.to_numeric(outer.valid_replicates,errors="coerce").min())
        axc.text(.01,-.20,f"{refits} valid refits with preprocessing + GMM + label alignment; original matching retained.",transform=axc.transAxes,color=COL["grey"],fontsize=6)
        panel(axc,"c")

        cross = fixed_diff.rename(columns={"latent_state":"state", "difference_dm_minus_non_dm":"mimic_difference"}).merge(
            eicu_diff.rename(columns={"transported_state":"state", "difference_dm_minus_non_dm":"eicu_difference"}),
            on=["state","next_state"], how="inner")
        cross["concordant"] = np.sign(cross.mimic_difference).eq(np.sign(cross.eicu_difference))
        x = 100 * cross.mimic_difference.to_numpy(); yy = 100 * cross.eicu_difference.to_numpy()
        lim = max(abs(x).max(), abs(yy).max()) * 1.08
        axd.axhline(0,color="#D0D0D0",lw=.7); axd.axvline(0,color="#D0D0D0",lw=.7)
        axd.plot([-lim,lim],[-lim,lim],ls="--",lw=.8,color="#A8A8A8")
        for ok,c in [(True,COL["teal"]),(False,COL["red"])]:
            q=cross[cross.concordant.eq(ok)]
            axd.scatter(100*q.mimic_difference,100*q.eicu_difference,s=24,color=c,alpha=.85,
                        label="Direction concordant" if ok else "Discordant")
        for state,label in [("Stable metabolic","Stable→hyper"),("Renal-metabolic vulnerability","Renal→hyper")]:
            q=cross[(cross.state.eq(state)) & (cross.next_state.eq("Hyperglycaemic instability"))]
            if len(q):
                r=q.iloc[0]; axd.annotate(label,(100*r.mimic_difference,100*r.eicu_difference),xytext=(4,4),textcoords="offset points",fontsize=5.5)
        axd.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel="MIMIC DM − non-DM transition (pp)",ylabel="eICU DM − non-DM transition (pp)",title="External clinical-state dynamic replication")
        axd.set_aspect("equal", adjustable="box")
        axd.legend(fontsize=5.5,loc="upper left")
        rep=paired_replication_summary(x,yy)
        axd.text(.98,.03,f"{rep['n']} edges: Pearson r={rep['pearson']:.3f} [edge-bootstrap {rep['pearson_lo']:.3f}, {rep['pearson_hi']:.3f}]\nSpearman r={rep['spearman']:.3f}; sign={100*rep['sign']:.1f}%",transform=axd.transAxes,ha="right",va="bottom",fontsize=6,color=COL["grey"])
        panel(axd,"d")
        fig.suptitle("Diabetes is associated with excess entry into hyperglycaemic-instability states", fontsize=11, y=.995)
        save(fig,"Figure_3_dynamic_diabetes_metabolic_landscape")


    def fig4():
        perf=pd.read_csv(PPD_FIXED/"fixed_time_performance.csv")
        base=pd.read_csv(CLIN/"cam_simple_baseline_comparison_summary.csv")
        geom=pd.read_csv(PPD/"domain_applicability_internal_geometry_test.csv")
        cam=pd.read_csv(PPD/"ppd_state_strict_cam_adjusted_bootstrap_test.csv")
        landmark_label=fixed_time_axis_label()
        fig=plt.figure(figsize=(7.1,6.7)); gs=fig.add_gridspec(2,2,wspace=.36,hspace=.50)
        axa,axb,axc,axd=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

        q=perf[perf.analysis_scope.eq("landmark_eligible_valid_icu_anchor")]
        variants=["temporal_only_no_text","physiology_only_temporal","physiology_plus_static_los_masked","static_temporal_no_text_los_masked"]
        labels=["Temporal, no text","Physiology only","Physiology + static","Static + temporal"]
        colors=[COL["navy"],COL["teal"],COL["orange"],COL["purple"]]
        for v,l,c in zip(variants,labels,colors):
            z=q[q.variant.eq(v)].sort_values("horizon_hours")
            axa.plot(z.horizon_hours,z.macro_auprc,marker="o",ms=3,label=l,color=c,lw=1.4)
        ymin=float(q.macro_auprc.min()-0.025); ymax=float(q.macro_auprc.max()+0.012)
        axa.set_ylim(ymin,ymax)
        axa.set(xticks=[6,12,24,48],xlabel=landmark_label,ylabel="Six-task macro-AUPRC",title="Frozen fixed-time PPD-EHR performance")
        axa.legend(fontsize=5.8,ncol=2,loc="lower right")
        n=q[q.variant.eq("physiology_only_temporal")].sort_values("horizon_hours")
        for _,r in n.iterrows(): axa.text(r.horizon_hours,ymin+.004,f"n={int(r.n):,}",ha="center",va="bottom",fontsize=5.3,color=COL["grey"])
        panel(axa,"a")

        names={"severity_plus_total_calories":"Severity + total calories","plus_nutrition_instability":"+ nutrition instability","plus_latent_state":"+ latent state"}
        b=base.set_index("model").loc[list(names)].reset_index(); yy=np.arange(len(b))[::-1]
        axb.barh(yy,b.AUPRC_mean,color=[COL["grey"],COL["orange"],COL["teal"]],height=.55)
        span=max(float(b.AUPRC_mean.max()-b.AUPRC_mean.min()),.005)
        axb.set_xlim(float(b.AUPRC_mean.min()-0.6*span),float(b.AUPRC_mean.max()+0.6*span))
        axb.set_yticks(yy,[names[x] for x in b.model]); axb.set_xlabel("Strict CAM cross-validated AUPRC")
        axb.set_title("Clinical-state baseline comparison (not PPD states)")
        for y,v in zip(yy,b.AUPRC_mean): axb.text(v+.0005,y,f"{v:.4f}",va="center",fontsize=6)
        instability_delta=float(b.loc[b.model.eq("plus_nutrition_instability"),"delta_AUPRC_vs_total_calories"].iloc[0])
        state_delta=float(b.loc[b.model.eq("plus_latent_state"),"delta_AUPRC_vs_total_calories"].iloc[0])
        axb.text(.01,-.20,f"Clinical features: instability ΔAUPRC {instability_delta:+.5f}; + clinical latent state {state_delta:+.5f}.",transform=axb.transAxes,fontsize=6,color=COL["grey"])
        panel(axb,"b")

        target_order=["clinical_timewindow_linked","separate_admin_source_linked","strict_cam_analysis"]
        lab={"clinical_timewindow_linked":"Clinical linkage","separate_admin_source_linked":"Medication-source linkage","strict_cam_analysis":"Strict CAM subset (48 h)"}
        g=geom[geom.target_subset.isin(target_order)].copy()
        for target,c in zip(target_order,[COL["navy"],COL["orange"],COL["red"]]):
            z=g[g.target_subset.eq(target)].sort_values("horizon_hours")
            axc.plot(z.horizon_hours,z.kde_overlap_pc1_to_pc5_mean,marker="o",ms=3,color=c,label=lab[target])
        axc.set(xticks=[6,12,24,48],xlabel=landmark_label,ylabel="PCA–KDE density overlap",title="Validation-referenced domain applicability")
        axc.legend(fontsize=5.7, loc="lower left")
        axc.text(.01,-.22,"No model-ready non-DM sparse-event bundle: DM/non-DM PPD transport not estimable.",transform=axc.transAxes,fontsize=5.8,color=COL["grey"])
        panel(axc,"c")

        state_order=cam.sort_values("standardized_cam_risk",ascending=True)
        y=np.arange(len(state_order)); v=state_order.standardized_cam_risk.to_numpy(); lo=state_order.ci_lower.to_numpy(); hi=state_order.ci_upper.to_numpy()
        axd.errorbar(v,y,xerr=[v-lo,hi-v],fmt="o",color=COL["orange"],ecolor=COL["orange"],capsize=2,lw=1.1)
        axd.set_yticks(y,[x.replace(" multisystem state "," ").replace(" vulnerability","") for x in state_order.ppd_state])
        axd.set_xlabel("Standardized strict CAM risk")
        axd.set_title("Absolute test-set CAM risk by validation-derived state")
        cam_n = int(cam.n.sum()) if len(cam) and "n" in cam else 0
        cam_events = int(cam.events.sum()) if len(cam) and "events" in cam else 0
        global_p=float(cam.global_state_lrt_p.iloc[0]) if len(cam) and "global_state_lrt_p" in cam else np.nan
        reference=str(cam.reference_state.iloc[0]) if len(cam) else "not available"
        axd.text(.01,-.30,"Validation-derived PPD state system (orange; distinct from Fig. 3 clinical states).\n"
                 f"Held-out test: n={cam_n:,}, events={cam_events:,}; global state LRT P={global_p:.3g}.\n"
                 f"Validation-selected reference for supplementary RRs: {reference}.",
                 transform=axd.transAxes,fontsize=5.7,color=COL["grey"])
        panel(axd,"d")
        fig.suptitle("Frozen physiological representations support dynamic risk estimation with bounded CAM anchoring",fontsize=11,y=.995)
        save(fig,"Figure_4_ppd_linkage_baseline_domain")


    def fig5():
        summary=pd.read_csv(OVERLAP/"propofol_treatment_aware_energy_summary.csv")
        rows=pd.read_csv(OVERLAP/"propofol_exposed_treatment_aware_energy_rows.csv")
        eff=pd.read_csv(CLIN/"nutrition_steroid_transition_effects.csv")
        ppd=pd.read_csv(PPD/"ppd_intervention_conditioned_transition_effects_test.csv")
        med=pd.read_csv(CLIN/"latent_state_mediation_bootstrap.csv")
        fig=plt.figure(figsize=(7.2,6.4)); gs=fig.add_gridspec(2,2,wspace=.42,hspace=.55)
        axa,axb,axc,axd=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

        share_col=next(c for c in rows.columns if "share" in c.lower() and "propofol" in c.lower())
        vals=100*pd.to_numeric(rows[share_col],errors="coerce").dropna()
        vals=vals[(vals>=0)&(vals<=100)]
        axa.hist(vals,bins=np.arange(0,75,5),color=COL["teal"],edgecolor="white")
        medv=float(vals.median()); axa.axvline(medv,color=COL["orange"],ls="--",lw=1.4)
        axa.text(medv+1,axa.get_ylim()[1]*.88,f"median {medv:.1f}%",color=COL["orange"],fontweight="bold")
        axa.set(xlabel="Propofol-derived share of treatment-aware energy (%)",ylabel="ICU stays",title="Drug-derived energy changes exposure definition")
        s=summary.iloc[0]
        recorded=100*float(s.recorded_energy20_rate_propofol_with_weight)
        aware=100*float(s.treatment_aware_energy20_rate_propofol_with_weight)
        reclassified=int(float(s.reclassified_by_propofol_n))
        axa.text(.01,-.22,f"Recorded energy threshold: {recorded:.1f}% → {aware:.1f}%; {reclassified:,} stays reclassified.",transform=axa.transAxes,fontsize=6,color=COL["grey"])
        panel(axa,"a")

        terms=[("steroid_any","Steroid on/off"),("insulin_response_residual_z","Insulin response residual"),("nutrition_z","Nutrition instability"),("nutrition_x_steroid","Nutrition × steroid"),("treatment_aware_instability_z","Treatment-aware instability"),("negative_control","Negative control"),("propofol_kcal_z","Propofol context")]
        rr=[]
        for term,label in terms:
            z=eff[eff.term.eq(term)]
            if len(z):
                r=z.iloc[0]; rr.append((label,r.odds_ratio,r.ci_lower,r.ci_upper))
        y=np.arange(len(rr))[::-1]; v=np.array([x[1] for x in rr]); lo=np.array([x[2] for x in rr]); hi=np.array([x[3] for x in rr])
        axb.axvline(1,color="#B8B8B8",lw=.8); axb.errorbar(v,y,xerr=[v-lo,hi-v],fmt="o",color=COL["navy"],ecolor=COL["navy"],capsize=2,lw=1.1)
        axb.set_yticks(y,[x[0] for x in rr]); axb.set_xlabel("OR for transition to renal–metabolic vulnerability"); axb.set_title("Clinical-state candidate associations")
        axb.text(.01,-.22,"Observational associations; adjusted for severity and current state.",transform=axb.transAxes,fontsize=6,color=COL["grey"])
        panel(axb,"b")

        pp=ppd.copy(); pp["label"]=pp.term.map({"nutrition_z":"Nutrition instability","steroid_any":"Steroid on/off","nutrition_x_steroid":"Nutrition × steroid","insulin_residual_z":"Insulin response residual","insulin_units":"Insulin units","propofol_z":"Propofol context"})
        pp=pp[pp.label.notna()]
        y=np.arange(len(pp))[::-1]; v=pp.odds_ratio.to_numpy(); lo=pp.ci_lower.to_numpy(); hi=pp.ci_upper.to_numpy()
        axc.axvline(1,color="#B8B8B8",lw=.8); axc.errorbar(v,y,xerr=[v-lo,hi-v],fmt="o",color=COL["orange"],ecolor=COL["orange"],capsize=2,lw=1.1)
        axc.set_yticks(y,pp.label); axc.set_xlabel("OR for transition to PPD hyperglycaemic state"); axc.set_title("Stay-level held-out test PPD candidate associations")
        pp_all_cross=bool(((pp.ci_lower<=1)&(pp.ci_upper>=1)).all()) if len(pp) else True
        pp_ci_text="all displayed 95% CIs cross 1" if pp_all_cross else "at least one displayed 95% CI excludes 1"
        axc.text(.01,-.31,"Validation-derived PPD states; held-out-test, target-state-specific associations.\n"
                 f"{pp_ci_text}; exposure-support complete-case analysis.",
                 transform=axc.transAxes,fontsize=5.7,color=COL["grey"])
        panel(axc,"c")

        mi=med[med.estimand.eq("latent_state_indirect_effect")].copy()
        mi["label"]=mi.exposure.map({"steroid_any":"Steroid","high_nutrition_instability":"High nutrition instability","high_insulin_response_residual":"High insulin residual"})
        y=np.arange(len(mi))[::-1]; v=100*mi.risk_difference.to_numpy(); lo=100*mi.ci_lower.to_numpy(); hi=100*mi.ci_upper.to_numpy()
        axd.axvline(0,color="#B8B8B8",lw=.8); axd.errorbar(v,y,xerr=[v-lo,hi-v],fmt="o",color=COL["red"],ecolor=COL["red"],capsize=2,lw=1.1)
        axd.set_yticks(y,mi.label); axd.set_xlabel("Latent-state indirect risk difference (percentage points)"); axd.set_title("Formal mediation was not confirmed")
        med_all_cross=bool(((mi.ci_lower<=0)&(mi.ci_upper>=0)).all()) if len(mi) else True
        med_ci_text="all patient-bootstrap 95% CIs cross 0" if med_all_cross else "at least one patient-bootstrap 95% CI excludes 0"
        med_n=int(pd.to_numeric(mi.n,errors="coerce").max()) if len(mi) else 0
        med_events=int(pd.to_numeric(mi.events,errors="coerce").max()) if len(mi) else 0
        axd.text(.01,-.22,f"{med_ci_text}; n={med_n:,}, events={med_events:,}.",transform=axd.transAxes,fontsize=6,color=COL["grey"])
        panel(axd,"d")
        fig.suptitle("Treatment-conditioned analyses nominate candidate pathways without confirming mediation",fontsize=11,y=.995)
        save(fig,"Figure_5_candidate_intervention_conditioned_pathways")


    def supp_external_transport():
        mimic_diff = pd.read_csv(CLIN / "state_transition_differences_dm_minus_nondm.csv")
        eicu_diff = pd.read_csv(EICU_TRANSPORT / "eicu_state_transition_differences_dm_minus_nondm.csv")
        mimic_occ = pd.read_csv(CLIN / "state_occupancy_dm_vs_nondm.csv")
        eicu_occ = pd.read_csv(EICU_TRANSPORT / "eicu_state_occupancy_by_dm.csv")
        prof = pd.read_csv(EICU_TRANSPORT / "mimic_eicu_state_profile_replication.csv")
        geom = json.loads((EICU_TRANSPORT / "clinical_state_transport_geometry.json").read_text())
        order = ["Stable metabolic", "Hyperglycaemic instability", "Renal-metabolic vulnerability",
                 "Inflammatory-haemodynamic stress", "Mixed vulnerability 1", "Mixed vulnerability 2"]
        fig=plt.figure(figsize=(7.2,7.0)); gs=fig.add_gridspec(2,2,wspace=.42,hspace=.78)
        axa,axb,axc,axd=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]

        mm=mimic_diff.pivot(index="latent_state",columns="next_state",values="difference_dm_minus_non_dm").reindex(index=order,columns=order)*100
        ee=eicu_diff.pivot(index="transported_state",columns="next_state",values="difference_dm_minus_non_dm").reindex(index=order,columns=order)*100
        lim=max(np.abs(mm.to_numpy()).max(),np.abs(ee.to_numpy()).max())
        cmap=sns.diverging_palette(230,20,as_cmap=True)
        for ax,mat,title,label in [(axa,mm,"MIMIC transition differences","a"),(axb,ee,"eICU transported-state differences","b")]:
            sns.heatmap(mat,ax=ax,cmap=cmap,center=0,vmin=-lim,vmax=lim,annot=True,fmt=".1f",annot_kws={"fontsize":5.2},cbar=False,linewidths=.45,linecolor="white")
            ax.set_yticklabels([STATE_SHORT[x] for x in order],rotation=0)
            ax.set_xticklabels([STATE_SHORT[x] for x in order],rotation=45,ha="right")
            ax.set(xlabel="Next state",ylabel="Current state",title=f"{title} (scale ±{lim:.1f} pp)")
            panel(ax,label)

        p=prof.set_index("state").reindex(order).reset_index().sort_values("profile_z_pearson_r")
        y=np.arange(len(p)); axc.hlines(y,0,p.profile_z_pearson_r,color=COL["light"],lw=4)
        axc.scatter(p.profile_z_pearson_r,y,color=COL["teal"],s=34)
        axc.set_yticks(y,[STATE_SHORT[x].replace("\n"," ") for x in p.state]); axc.set_xlim(0,1)
        axc.set(xlabel="MIMIC–eICU standardized profile correlation",title="State phenotype transport")
        axc.text(.02,.04,f"Mean profile r={p.profile_z_pearson_r.mean():.3f}\nKDE overlap={geom['kde_overlap_pc1_to_pc5_mean']:.3f}\nDataset AUROC={geom['dataset_classifier_auroc']:.3f}",transform=axc.transAxes,fontsize=6,color=COL["grey"])
        panel(axc,"c")

        mo=mimic_occ[mimic_occ.bin.le(7)].rename(columns={"latent_state":"state","proportion":"mimic_prop"})
        eo=eicu_occ.rename(columns={"transported_state":"state","proportion":"eicu_prop"})
        q=mo[["diabetes","bin","state","mimic_prop"]].merge(eo[["diabetes","bin","state","eicu_prop"]],on=["diabetes","bin","state"])
        for dm,c,l in [(0,COL["grey"],"non-DM"),(1,COL["red"],"DM")]:
            z=q[q.diabetes.eq(dm)]; axd.scatter(z.mimic_prop,z.eicu_prop,s=20,color=c,alpha=.7,label=l)
        lim2=max(q.mimic_prop.max(),q.eicu_prop.max())*1.05; axd.plot([0,lim2],[0,lim2],ls="--",color="#A8A8A8",lw=.8)
        axd.set(xlim=(0,lim2),ylim=(0,lim2),xlabel="MIMIC state occupancy",ylabel="eICU state occupancy",title="State–time occupancy replication")
        axd.set_aspect("equal",adjustable="box"); axd.legend(fontsize=6)
        occ_rep=paired_replication_summary(q.mimic_prop,q.eicu_prop)
        axd.text(.97,.04,f"{occ_rep['n']} cells: Pearson r={occ_rep['pearson']:.3f}\nsign concordance={100*occ_rep['sign']:.1f}%",transform=axd.transAxes,ha="right",fontsize=6,color=COL["grey"])
        panel(axd,"d")
        fig.suptitle("External transport supports the direction of diabetes-associated clinical-state dynamics",fontsize=11,y=.995)
        save(fig,"Supplementary_Figure_external_clinical_state_transport")


    def export_source_data():
        source=OUT/"source_data"
        source.mkdir(parents=True,exist_ok=True)
        tables={
            "Figure_3a_clinical_state_profiles.csv": pd.read_csv(CLIN/"latent_state_profiles.csv"),
            "Figure_3b_state_occupancy_dm_vs_nondm.csv": pd.read_csv(CLIN/"state_occupancy_dm_vs_nondm.csv"),
            "Figure_3c_outer_state_bootstrap.csv": pd.read_csv(CLIN/"outer_state_bootstrap_summary.csv"),
            "Figure_3d_mimic_transition_edges.csv": pd.read_csv(CLIN/"state_transition_differences_dm_minus_nondm.csv"),
            "Figure_3d_eicu_transition_edges.csv": pd.read_csv(EICU_TRANSPORT/"eicu_state_transition_differences_dm_minus_nondm.csv"),
            "Figure_4a_fixed_time_performance_test.csv": pd.read_csv(PPD_FIXED/"fixed_time_performance.csv"),
            "Figure_4b_clinical_state_cam_baseline.csv": pd.read_csv(CLIN/"cam_simple_baseline_comparison_summary.csv"),
            "Figure_4c_domain_geometry_test.csv": pd.read_csv(PPD/"domain_applicability_internal_geometry_test.csv"),
            "Figure_4d_ppd_state_cam_absolute_risk_test.csv": pd.read_csv(PPD/"ppd_state_strict_cam_adjusted_bootstrap_test.csv"),
            "Supplement_subject_disjoint_cam.csv": pd.read_csv(PPD/"ppd_state_strict_cam_adjusted_bootstrap_test_subject_disjoint.csv"),
            "Supplement_subject_disjoint_treatment_effects.csv": pd.read_csv(PPD/"ppd_intervention_conditioned_transition_effects_test_subject_disjoint.csv"),
            "Supplement_intervention_exposure_support.csv": pd.read_csv(PPD/"ppd_intervention_exposure_support_audit.csv"),
        }
        manifest=[]
        for name,table in tables.items():
            path=source/name; table.to_csv(path,index=False)
            manifest.append({"file":name,"rows":len(table),"columns":len(table.columns),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        pd.DataFrame(manifest).to_csv(source/"source_data_manifest.csv",index=False)


    def main():
        global CLIN, PPD_FIXED, PPD, EICU_TRANSPORT, OVERLAP, OUT
        args = parse_args()
        CLIN = args.clinical_root
        PPD_FIXED = args.fixed_test_dir
        PPD = args.ppd_downstream_dir
        EICU_TRANSPORT = args.eicu_transport_dir
        OVERLAP = args.overlap_dir
        OUT = args.output_dir
        OUT.mkdir(parents=True, exist_ok=True)
        # The longitudinal figure and DAG are generated by make_longitudinal_figures.py.
        # The former treatment-conditioned Figure 5 remains a legacy exploratory
        # function and is not emitted by the final figure build.
        fig3(); fig4(); supp_external_transport(); export_source_data()
        manifest=[]
        for f in sorted(OUT.iterdir()): manifest.append({"file":f.name,"bytes":f.stat().st_size})
        pd.DataFrame(manifest).to_csv(OUT/"figure_export_manifest.csv",index=False)

    return SimpleNamespace(**locals())

_latent_plot = _load_latent_plot()


# ==============================================================================
# State-occupancy figure
# ==============================================================================

def _load_occupancy_plot():
    """Build current Figure 5 from six-state temporal occupancy data.

    The figure is deliberately cohort-level: it shows occupancy of the six
    clinical states over 6-hour windows and the supplied adjacent-window
    transition probabilities into the renal-metabolic state. No model is
    retrained and no patient-level trajectory is inferred.
    """
    from pathlib import Path
    import shutil

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd


    ROOT = Path(__file__).resolve().parents[2]
    FIG_ROOT = ROOT / "outputs" / "figure5_state_occupancy_trajectory"
    INDIVIDUAL = FIG_ROOT / "individual_panels"
    DATA_OUT = FIG_ROOT / "source_data"

    SOURCE_DATA = ROOT / "data" / "derived" / "figure5_state_transport"

    STATE_ORDER = [
        "Stable metabolic",
        "Renal-metabolic vulnerability",
        "Mixed vulnerability 1",
        "Mixed vulnerability 2",
        "Inflammatory-haemodynamic stress",
        "Hyperglycaemic instability",
    ]
    STATE_LABELS = {
        "Stable metabolic": "Stable\nmetabolic",
        "Renal-metabolic vulnerability": "Renal-metabolic\nvulnerability",
        "Mixed vulnerability 1": "Mixed\nvulnerability 1",
        "Mixed vulnerability 2": "Mixed\nvulnerability 2",
        "Inflammatory-haemodynamic stress": "Inflammatory-\nhaemodynamic stress",
        "Hyperglycaemic instability": "Hyperglycaemic\ninstability",
    }
    STATE_COLORS = {
        "Stable metabolic": "#5B9BD5",
        "Renal-metabolic vulnerability": "#F2A65A",
        "Mixed vulnerability 1": "#8D5AA8",
        "Mixed vulnerability 2": "#C7B9D6",
        "Inflammatory-haemodynamic stress": "#A8C9B2",
        "Hyperglycaemic instability": "#B85C6A",
    }
    COHORT_COLORS = {"MIMIC-IV": "#5B9BD5", "eICU": "#B85C6A"}
    INK = "#1F2A44"
    MUTED = "#5E6C76"
    GRID = "#D9E1E6"


    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


    def style_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK)
            ax.spines[side].set_linewidth(0.75)
        ax.tick_params(which="both", length=2.5, width=0.65, color=MUTED)
        ax.tick_params(axis="both", labelcolor=INK)
        if grid_axis:
            ax.grid(axis=grid_axis, color=GRID, lw=0.5, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)


    def save_pub(fig: plt.Figure, stem: str) -> None:
        FIG_ROOT.mkdir(parents=True, exist_ok=True)
        INDIVIDUAL.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "pdf", "png", "tiff"):
            for base in (FIG_ROOT, INDIVIDUAL):
                path = base / f"{stem}.{ext}"
                kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
                if ext in ("png", "tiff"):
                    kwargs["dpi"] = 600
                fig.savefig(path, **kwargs)
        plt.close(fig)


    def aggregate_occupancy(path: Path, cohort: str, state_col: str, weight_col: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        weighted = raw.assign(weighted_count=raw[weight_col].astype(float))
        grouped = weighted.groupby(["bin", state_col], as_index=False)["weighted_count"].sum()
        totals = grouped.groupby("bin", as_index=False)["weighted_count"].sum().rename(columns={"weighted_count": "bin_total"})
        grouped = grouped.merge(totals, on="bin", how="left")
        grouped["proportion"] = grouped["weighted_count"] / grouped["bin_total"]
        grouped["hours"] = grouped["bin"].astype(float) * 6.0 + 3.0
        grouped["cohort"] = cohort
        grouped = grouped.rename(columns={state_col: "latent_state"})
        return grouped[["cohort", "bin", "hours", "latent_state", "weighted_count", "bin_total", "proportion"]]


    def renal_transition_table(path: Path, cohort: str, state_col: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        raw = raw.rename(columns={state_col: "latent_state"})
        out = raw.loc[raw["next_state"].eq("Renal-metabolic vulnerability"), ["latent_state", "p_dm", "p_non_dm"]].copy()
        out = out.melt(id_vars="latent_state", var_name="diabetes_group", value_name="probability")
        out["diabetes_group"] = out["diabetes_group"].map({"p_dm": "DM", "p_non_dm": "non-DM"})
        out["cohort"] = cohort
        return out[["cohort", "latent_state", "diabetes_group", "probability"]]


    def build_figure(occupancy: pd.DataFrame, transitions: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(12.2, 6.35))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.08], wspace=0.48, left=0.085, right=0.98, top=0.80, bottom=0.30)

        # Occupancy trajectories: state colour identifies the state and line style
        # identifies the cohort. eICU ends at the last supplied 6-h window.
        ax = fig.add_subplot(gs[0, 0])
        style_axis(ax, "both")
        for cohort, linestyle, alpha in [("MIMIC-IV", "-", 1.0), ("eICU", "--", 0.78)]:
            sub = occupancy.loc[occupancy["cohort"].eq(cohort)]
            for state in STATE_ORDER:
                line = sub.loc[sub["latent_state"].eq(state)].sort_values("hours")
                ax.plot(
                    line["hours"],
                    line["proportion"],
                    color=STATE_COLORS[state],
                    lw=1.45 if cohort == "MIMIC-IV" else 1.15,
                    ls=linestyle,
                    alpha=alpha,
                    marker="o" if cohort == "MIMIC-IV" else None,
                    ms=2.8,
                    zorder=3,
                )
        ax.axvspan(48, 72, color=GRID, alpha=0.18, zorder=0)
        ax.text(60, 0.61, "eICU\nwindow ends", ha="center", va="top", fontsize=8.0, color=MUTED)
        ax.set_xlim(0, 72)
        ax.set_ylim(0, 0.68)
        ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_xlabel("Hours from ICU admission")
        ax.set_ylabel("State occupancy (proportion of 6-h windows)")
        ax.set_title("Six-state occupancy trajectories", loc="left", color=INK, fontweight="bold", pad=28)
        ax.text(0, 1.035, "Weighted cohort summaries; colours identify states, line styles identify cohorts", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, color=MUTED)
        ax.text(-0.07, 1.18, "c", transform=ax.transAxes, fontsize=12.0, fontweight="bold", color=INK)

        state_handles = [Line2D([0], [0], color=STATE_COLORS[s], lw=2.0, label=s.replace(" vulnerability", "\nvulnerability").replace(" stress", "\nstress")) for s in STATE_ORDER]
        cohort_handles = [
            Line2D([0], [0], color=INK, lw=1.5, ls="-", label="MIMIC-IV"),
            Line2D([0], [0], color=INK, lw=1.5, ls="--", label="eICU"),
        ]
        leg1 = ax.legend(handles=state_handles, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=3, fontsize=7.5, handlelength=1.55, columnspacing=1.0, handletextpad=0.4)
        ax.add_artist(leg1)
        ax.legend(handles=cohort_handles, loc="upper right", bbox_to_anchor=(1.0, -0.18), ncol=2, fontsize=7.6, handlelength=1.8, columnspacing=1.0, handletextpad=0.4)

        # Renal convergence: each source state's conditional probability of moving
        # into the renal-metabolic state in the next observed window.
        ax = fig.add_subplot(gs[0, 1])
        style_axis(ax, "x")
        y = np.arange(len(STATE_ORDER))[::-1]
        ymap = dict(zip(STATE_ORDER, y))
        ax.axhspan(ymap["Renal-metabolic vulnerability"] - 0.38, ymap["Renal-metabolic vulnerability"] + 0.38, color="#F2A65A", alpha=0.12, zorder=0)
        point_offsets = {
            ("MIMIC-IV", "DM"): 0.22,
            ("MIMIC-IV", "non-DM"): 0.08,
            ("eICU", "DM"): -0.08,
            ("eICU", "non-DM"): -0.22,
        }
        point_markers = {"DM": "o", "non-DM": "D"}
        for state, yy in ymap.items():
            for (cohort, diabetes_group), offset in point_offsets.items():
                row = transitions.loc[
                    transitions["cohort"].eq(cohort)
                    & transitions["latent_state"].eq(state)
                    & transitions["diabetes_group"].eq(diabetes_group)
                ]
                probability = float(row["probability"].iloc[0])
                ax.scatter(
                    probability,
                    yy + offset,
                    s=25,
                    color=COHORT_COLORS[cohort] if diabetes_group == "DM" else "white",
                    edgecolor=COHORT_COLORS[cohort],
                    marker=point_markers[diabetes_group],
                    linewidth=0.8,
                    zorder=3,
                )
                if state == "Renal-metabolic vulnerability":
                    ax.text(probability + 0.008, yy + offset, f"{probability:.1%}", fontsize=7.0, color=COHORT_COLORS[cohort], va="center")
        ax.set_xlim(0, 0.45)
        ax.set_ylim(-0.6, len(STATE_ORDER) - 0.4)
        ax.set_yticks(y, [STATE_LABELS[s] for s in STATE_ORDER])
        ax.set_xticks([0, 0.10, 0.20, 0.30, 0.40], ["0", "10%", "20%", "30%", "40%"])
        ax.set_xlabel("P(next state = renal-metabolic vulnerability)")
        ax.set_title("Renal convergence in adjacent-window transitions", loc="left", color=INK, fontweight="bold", pad=28)
        ax.text(0, 1.035, "DM and non-DM estimates shown separately", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, color=MUTED)
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COHORT_COLORS["MIMIC-IV"], markeredgecolor=COHORT_COLORS["MIMIC-IV"], markersize=5.0, label="MIMIC-IV DM"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COHORT_COLORS["MIMIC-IV"], markersize=4.7, label="MIMIC-IV non-DM"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=COHORT_COLORS["eICU"], markeredgecolor=COHORT_COLORS["eICU"], markersize=5.0, label="eICU DM"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COHORT_COLORS["eICU"], markersize=4.7, label="eICU non-DM"),
            ],
            loc="lower right",
            ncol=2,
            fontsize=7.0,
            columnspacing=0.8,
            handletextpad=0.3,
            borderaxespad=0.1,
        )

        fig.suptitle("Six clinical states shift over time, with recurrent cohort-level convergence on renal-metabolic vulnerability", x=0.085, y=0.96, ha="left", fontsize=13.0, fontweight="bold", color=INK)
        fig.text(0.085, 0.035, "Occupancy is aggregated across diabetes strata from weighted 6-h windows. Transition points are supplied DM/non-DM adjacent-window conditional probabilities; eICU data are available through 48 h.", ha="left", va="bottom", fontsize=7.5, color=MUTED)
        save_pub(fig, "Figure5_state_occupancy_transition_trajectory")


    def write_outputs() -> None:
        FIG_ROOT.mkdir(parents=True, exist_ok=True)
        DATA_OUT.mkdir(parents=True, exist_ok=True)
        source_map = {
            "mimic_state_occupancy.csv": "mimic_state_occupancy.csv",
            "eicu_state_occupancy.csv": "eicu_state_occupancy.csv",
            "mimic_transition_edges.csv": "mimic_transition_edges.csv",
            "eicu_transition_edges.csv": "eicu_transition_edges.csv",
        }
        for source_name, target_name in source_map.items():
            shutil.copy2(SOURCE_DATA / source_name, DATA_OUT / target_name)


    def main() -> None:
        write_outputs()
        mimic_occ = aggregate_occupancy(SOURCE_DATA / "mimic_state_occupancy.csv", "MIMIC-IV", "latent_state", "weighted_windows")
        eicu_occ = aggregate_occupancy(SOURCE_DATA / "eicu_state_occupancy.csv", "eICU", "transported_state", "weighted_n")
        occupancy = pd.concat([mimic_occ, eicu_occ], ignore_index=True)
        mimic_trans = renal_transition_table(SOURCE_DATA / "mimic_transition_edges.csv", "MIMIC-IV", "latent_state")
        eicu_trans = renal_transition_table(SOURCE_DATA / "eicu_transition_edges.csv", "eICU", "transported_state")
        transitions = pd.concat([mimic_trans, eicu_trans], ignore_index=True)
        occupancy.to_csv(DATA_OUT / "Figure5_state_occupancy_aggregated.csv", index=False)
        transitions.to_csv(DATA_OUT / "Figure5_renal_transition_probabilities.csv", index=False)
        build_figure(occupancy, transitions)
        (FIG_ROOT / "README.md").write_text(
            """# Figure 5 six-state occupancy and renal-convergence figure

    This current Figure 5 version uses six clinical-state occupancy trajectories and adjacent-window transition evidence. MIMIC-IV is shown with solid lines and eICU with dashed lines. The eICU table ends at 48 h.

    The transition panel keeps the supplied DM and non-DM conditional probabilities separate. Occupancy and transition estimates are cohort-level observational summaries; they do not establish patient-level causal paths.

    Exports are available as SVG, PDF, PNG and 600-dpi TIFF in this folder and in `individual_panels`. Source tables and derived tidy tables are in `source_data`.
    """,
            encoding="utf-8",
        )
        print(f"Wrote Figure 5 state occupancy figure to {FIG_ROOT}")

    return SimpleNamespace(**locals())

_occupancy_plot = _load_occupancy_plot()


STAGES = {
    "latent-trajectories": _latent_plot.main,
    "state-occupancy": _occupancy_plot.main,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Clinical-state manuscript figures')
    parser.add_argument("stage", choices=STAGES)
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    stage = sys.argv[1]
    if stage not in STAGES:
        parser.error(f"invalid stage: {stage}")
    if stage == "state-occupancy" and any(arg in {"-h", "--help"} for arg in sys.argv[2:]):
        print("state-occupancy: builds the fixed manuscript state-occupancy figure (no extra options).")
        return
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    STAGES[stage]()


if __name__ == "__main__":
    main()
