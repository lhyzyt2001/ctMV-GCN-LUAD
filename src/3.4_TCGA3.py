from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.stats.multitest import multipletests

from config import ROBUSTNESS_DIR, TCGA_CLINICAL_FILE, TCGA_DIR, TCGA_EXPRESSION_FILE, ensure_result_dirs
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


def add_fdr(results: pd.DataFrame, p_column: str, output_column: str) -> pd.DataFrame:
    results[output_column] = np.nan
    valid = results[p_column].notna()
    if valid.any():
        results.loc[valid, output_column] = multipletests(
            results.loc[valid, p_column], method="fdr_bh"
        )[1]
    return results


def map_stage(value):
    if pd.isna(value):
        return np.nan
    text = str(value).upper().replace("STAGE", "").strip()
    if text.startswith("IV") or text.startswith("III"):
        return 1.0
    if text.startswith("II") or text.startswith("I"):
        return 0.0
    return np.nan


def prepare_survival_data():
    clinical = pd.read_csv(TCGA_CLINICAL_FILE, sep="\t")
    expression = pd.read_csv(TCGA_EXPRESSION_FILE, sep="\t", index_col=0)
    clinical["sampleID"] = clinical["sampleID"].astype(str)
    clinical = clinical[clinical["sampleID"].str[13:15].eq("01")].copy()
    clinical["patient"] = clinical["sampleID"].str[:12]
    clinical = clinical.sort_values("sampleID").drop_duplicates("patient", keep="first")
    survival = pd.DataFrame({
        "sample": clinical["sampleID"],
        "patient": clinical["patient"],
        "event": clinical["vital_status"].astype(str).str.upper().eq("DECEASED").astype(int),
    })
    death = pd.to_numeric(clinical["days_to_death"], errors="coerce")
    followup = pd.to_numeric(clinical["days_to_last_followup"], errors="coerce")
    survival["time_months"] = np.where(survival["event"].eq(1), death, followup) / 30.4375
    survival["age"] = pd.to_numeric(clinical.get("age_at_initial_pathologic_diagnosis"), errors="coerce")
    survival["gender"] = clinical.get("gender").astype(str).str.upper().map({"MALE": 1.0, "FEMALE": 0.0})
    survival["stage_high"] = clinical.get("pathologic_stage").map(map_stage)
    survival["new_tumor_event"] = clinical.get("new_tumor_event_after_initial_treatment").astype(str).str.upper().eq("YES").astype(int)
    new_event_days = pd.to_numeric(clinical.get("days_to_new_tumor_event_after_initial_treatment"), errors="coerce")
    survival["new_tumor_event_time_months"] = np.where(
        survival["new_tumor_event"].eq(1), new_event_days, followup
    ) / 30.4375
    survival = survival[(survival["time_months"] > 0) & survival["time_months"].notna()]
    return survival, expression


def km_step(ax, time, event, label, color):
    estimate = SurvfuncRight(np.asarray(time), np.asarray(event))
    x = np.r_[0.0, estimate.surv_times]
    y = np.r_[1.0, estimate.surv_prob]
    ax.step(x, y, where="post", label=label, color=color)
    censor_times = np.asarray(time)[np.asarray(event) == 0]
    if len(censor_times):
        positions = np.searchsorted(estimate.surv_times, censor_times, side="right") - 1
        censor_survival = np.where(positions >= 0, estimate.surv_prob[np.maximum(positions, 0)], 1.0)
        ax.plot(censor_times, censor_survival, linestyle="none", marker="+", markersize=3, markeredgewidth=0.6, color=color, alpha=0.55)


def make_km_plot(frame, gene, row):
    apply_bmc_style()
    fig = plt.figure(figsize=(FULL_WIDTH * 0.72, 4.25))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.0], hspace=0.16)
    ax = fig.add_subplot(grid[0])
    risk_ax = fig.add_subplot(grid[1])
    high = frame[gene] > frame[gene].median()
    km_step(ax, frame.loc[high, "time_months"], frame.loc[high, "event"], f"{gene} high (n={high.sum()})", "#D55E00")
    km_step(ax, frame.loc[~high, "time_months"], frame.loc[~high, "event"], f"{gene} low (n={(~high).sum()})", "#0072B2")
    max_time = min(240, int(np.nanmax(frame["time_months"])))
    ticks = np.arange(0, max_time + 1, 60)
    ax.set(xlabel="", ylabel="Survival probability", title=f"TCGA-LUAD: {gene}", xlim=(0, max_time), ylim=(0, 1.02), xticks=ticks)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(frameon=False)
    ax.grid(color="0.9", linewidth=0.5)
    annotation = f"Log-rank P={row.LogRank_P:.3g}, FDR={row.LogRank_FDR:.3g}\nAdjusted HR={row.Adjusted_HR:.2f} (95% CI {row.CI_Lower:.2f}–{row.CI_Upper:.2f})"
    ax.text(0.03, 0.05, annotation, transform=ax.transAxes, fontsize=7, va="bottom")
    risk_high = [int(((frame["time_months"] >= t) & high).sum()) for t in ticks]
    risk_low = [int(((frame["time_months"] >= t) & ~high).sum()) for t in ticks]
    risk_ax.axis("off")
    risk_ax.set_title("Overall survival (months)", fontsize=8, pad=2)
    table = risk_ax.table(cellText=[risk_high, risk_low], rowLabels=["High", "Low"], colLabels=[str(t) for t in ticks], cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 0.78])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.07)
    save_figure(fig, TCGA_DIR / f"KM_{gene}")


def analyze_new_tumor_event(merged: pd.DataFrame, top_genes: list[str]) -> pd.DataFrame:
    rows = []
    for gene in top_genes:
        frame = merged[["new_tumor_event_time_months", "new_tumor_event", "age", "gender", "stage_high", gene]].dropna().copy()
        frame = frame[frame["new_tumor_event_time_months"] > 0]
        if frame[gene].max() > 50:
            frame[gene] = np.log2(frame[gene] + 1)
        high = (frame[gene] > frame[gene].median()).astype(int)
        logrank_p = np.nan
        if high.nunique() == 2:
            _, logrank_p = survdiff(frame["new_tumor_event_time_months"], frame["new_tumor_event"], high)
        design = pd.DataFrame({
            "gene_z": zscore(frame[gene], nan_policy="omit"),
            "age_z": zscore(frame["age"], nan_policy="omit"),
            "gender": frame["gender"].to_numpy(),
            "stage_high": frame["stage_high"].to_numpy(),
        }, index=frame.index)
        try:
            fit = PHReg(
                frame["new_tumor_event_time_months"], design, status=frame["new_tumor_event"], ties="efron"
            ).fit(disp=0)
            gene_index = design.columns.get_loc("gene_z")
            hr = float(np.exp(fit.params[gene_index]))
            ci = np.exp(fit.conf_int()[gene_index])
            cox_p = float(fit.pvalues[gene_index])
        except Exception:
            hr, ci, cox_p = np.nan, (np.nan, np.nan), 1.0
        rows.append({
            "Gene": gene,
            "N": len(frame),
            "Events": int(frame["new_tumor_event"].sum()),
            "LogRank_P": float(logrank_p) if pd.notna(logrank_p) else np.nan,
            "Adjusted_Cox_P": cox_p,
            "Adjusted_HR": hr,
            "CI_Lower": float(ci[0]),
            "CI_Upper": float(ci[1]),
        })
    results = pd.DataFrame(rows)
    results = add_fdr(results, "LogRank_P", "LogRank_FDR")
    results = add_fdr(results, "Adjusted_Cox_P", "Adjusted_Cox_FDR")
    results.to_csv(TCGA_DIR / "TCGA_LUAD_robust_top20_new_tumor_event_results.csv", index=False, encoding="utf-8-sig")

    apply_bmc_style()
    ordered = results.sort_values("Adjusted_HR").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 5.2))
    y = np.arange(len(ordered))
    xerr = np.vstack([ordered["Adjusted_HR"] - ordered["CI_Lower"], ordered["CI_Upper"] - ordered["Adjusted_HR"]])
    ax.errorbar(ordered["Adjusted_HR"], y, xerr=xerr, fmt="o", color="#009E73", capsize=2)
    ax.axvline(1, color="0.5", linestyle="--")
    ax.set(
        yticks=y,
        yticklabels=ordered["Gene"],
        xlabel="Adjusted hazard ratio per 1 SD expression",
        title="TCGA-LUAD new-tumor-event associations",
    )
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, TCGA_DIR / "TCGA_LUAD_robust_top20_new_tumor_event_forest")
    return results


def main() -> None:
    ensure_result_dirs()
    for pattern in ("KM_*.png", "KM_*.pdf"):
        for stale_plot in TCGA_DIR.glob(pattern):
            stale_plot.unlink()
    candidates = pd.read_csv(ROBUSTNESS_DIR / "robust_top20.csv").sort_values("Robust_Candidate_Rank")
    survival, expression = prepare_survival_data()
    survival.to_csv(TCGA_DIR / "TCGA_primary_tumor_survival_cohort.csv", index=False, encoding="utf-8-sig")
    top_genes = [g for g in candidates["Gene_Symbol"] if g in expression.index]
    selected = candidates.loc[candidates["Gene_Symbol"].isin(top_genes), ["Robust_Candidate_Rank", "Gene_Symbol"]]
    selected.rename(columns={"Gene_Symbol": "Gene"}).to_csv(
        TCGA_DIR / "robust_top20_candidates.csv", index=False, encoding="utf-8-sig"
    )
    expr = expression.loc[top_genes].T.reset_index().rename(columns={"index": "sample"})
    merged = survival.merge(expr, on="sample", how="inner")
    rows, plot_frames = [], {}
    for gene in top_genes:
        frame = merged[["time_months", "event", "age", "gender", "stage_high", gene]].dropna().copy()
        if frame[gene].max() > 50:
            frame[gene] = np.log2(frame[gene] + 1)
        high = (frame[gene] > frame[gene].median()).astype(int)
        logrank_p = np.nan
        if high.nunique() == 2:
            _, logrank_p = survdiff(frame["time_months"], frame["event"], high)
        design = pd.DataFrame({
            "gene_z": zscore(frame[gene], nan_policy="omit"),
            "age_z": zscore(frame["age"], nan_policy="omit"),
            "gender": frame["gender"].to_numpy(),
            "stage_high": frame["stage_high"].to_numpy(),
        }, index=frame.index)
        try:
            fit = PHReg(frame["time_months"], design, status=frame["event"], ties="efron").fit(disp=0)
            gene_index = design.columns.get_loc("gene_z")
            hr = float(np.exp(fit.params[gene_index]))
            ci = np.exp(fit.conf_int()[gene_index])
            cox_p = float(fit.pvalues[gene_index])
        except Exception:
            hr, ci, cox_p = np.nan, (np.nan, np.nan), 1.0
        rows.append({"Gene": gene, "N": len(frame), "Events": int(frame["event"].sum()), "LogRank_P": float(logrank_p) if pd.notna(logrank_p) else np.nan, "Adjusted_Cox_P": cox_p, "Adjusted_HR": hr, "CI_Lower": float(ci[0]), "CI_Upper": float(ci[1]), "Median_Group_Status": "tested" if high.nunique() == 2 else "not tested: one group after median split"})
        if high.nunique() == 2:
            plot_frames[gene] = frame
    results = pd.DataFrame(rows)
    results = add_fdr(results, "LogRank_P", "LogRank_FDR")
    results = add_fdr(results, "Adjusted_Cox_P", "Adjusted_Cox_FDR")
    results.to_csv(TCGA_DIR / "TCGA_LUAD_robust_top20_survival_results.csv", index=False, encoding="utf-8-sig")
    for row in results.itertuples(index=False):
        if row.Gene in plot_frames:
            make_km_plot(plot_frames[row.Gene], row.Gene, row)
    apply_bmc_style()
    ordered = results.sort_values("Adjusted_HR").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 5.2))
    y = np.arange(len(ordered))
    xerr = np.vstack([ordered["Adjusted_HR"] - ordered["CI_Lower"], ordered["CI_Upper"] - ordered["Adjusted_HR"]])
    ax.errorbar(ordered["Adjusted_HR"], y, xerr=xerr, fmt="o", color="#0072B2", capsize=2)
    ax.axvline(1, color="0.5", linestyle="--")
    ax.set(yticks=y, yticklabels=ordered["Gene"], xlabel="Adjusted hazard ratio per 1 SD expression", title="TCGA-LUAD prognostic associations")
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, TCGA_DIR / "TCGA_LUAD_robust_top20_forest")
    new_event_results = analyze_new_tumor_event(merged, top_genes)
    (TCGA_DIR / "interpretation_note.txt").write_text(
        "The genes are the post hoc robustness-aware top 20 and were not preregistered or prespecified. Median expression cutoffs, age/sex/stage-adjusted Cox models, and Benjamini-Hochberg correction were applied to overall survival and the secondary new-tumor-event endpoint. TCGA associations are secondary clinical evidence and are not interpreted as functional validation of therapeutic target efficacy.\n",
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(new_event_results.to_string(index=False))


if __name__ == "__main__":
    main()
