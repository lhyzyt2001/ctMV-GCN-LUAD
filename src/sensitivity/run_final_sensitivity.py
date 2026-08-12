from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, zscore
from sklearn.metrics import average_precision_score
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1] if BASE.name == "sensitivity" and BASE.parent.name == "src" else BASE
RESULTS = REPO / "results"
OUT = RESULTS / "sensitivity"

TCGA_EXPRESSION = Path()
TCGA_COHORT = Path()
GSE_METADATA = Path()
GSE_EXPRESSION = Path()
OOF = Path()
NODE_LABELS = Path()
TOP20 = Path()
RAW_DATASET = Path()

PRIMARY = "ctMV-GCN (local attention + class-weighted CE)"
DEGREE = "Network-degree logistic"
BOOTSTRAPS = 2000
SEED = 42


def fdr(values: pd.Series) -> np.ndarray:
    adjusted = np.full(len(values), np.nan)
    valid = values.notna().to_numpy()
    if valid.any():
        adjusted[valid] = multipletests(values.loc[valid].to_numpy(), method="fdr_bh")[1]
    return adjusted


def ph_test(
    time: pd.Series,
    status: pd.Series,
    design: pd.DataFrame,
    cohort: str,
    endpoint: str,
    gene: str,
) -> list[dict]:
    fit = PHReg(time.to_numpy(), design.to_numpy(), status=status.to_numpy(), ties="efron").fit(disp=0)
    residuals = np.asarray(fit.schoenfeld_residuals, dtype=float)
    event_mask = status.to_numpy(dtype=int) == 1
    event_times = time.to_numpy(dtype=float)[event_mask]
    transformed_time = rankdata(event_times, method="average") / (len(event_times) + 1.0)
    rows = []
    for index, covariate in enumerate(design.columns):
        residual = residuals[event_mask, index]
        valid = np.isfinite(residual) & np.isfinite(transformed_time)
        if valid.sum() >= 3 and np.nanstd(residual[valid]) > 0:
            rho, p_value = pearsonr(residual[valid], transformed_time[valid])
        else:
            rho, p_value = np.nan, np.nan
        rows.append({
            "Cohort": cohort,
            "Endpoint": endpoint,
            "Gene": gene,
            "Covariate": covariate,
            "N": len(design),
            "Events": int(status.sum()),
            "Time_Transform": "event-time rank/(events+1)",
            "Schoenfeld_R": float(rho) if np.isfinite(rho) else np.nan,
            "PH_P": float(p_value) if np.isfinite(p_value) else np.nan,
        })
    return rows


def load_tcga() -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = pd.read_csv(TOP20).sort_values("Robust_Candidate_Rank")["Gene_Symbol"].astype(str).tolist()
    cohort = pd.read_csv(TCGA_COHORT)
    expression = pd.read_csv(TCGA_EXPRESSION, sep="\t", index_col=0)
    available = [gene for gene in genes if gene in expression.index]
    expr = expression.loc[available].T.reset_index().rename(columns={"index": "sample"})
    return cohort.merge(expr, on="sample", how="inner"), pd.DataFrame({"Gene": available})


def tcga_ph() -> pd.DataFrame:
    merged, genes = load_tcga()
    details = []
    endpoints = [
        ("overall_survival", "time_months", "event"),
        ("new_tumor_event", "new_tumor_event_time_months", "new_tumor_event"),
    ]
    for gene in genes["Gene"]:
        for endpoint, time_col, status_col in endpoints:
            columns = [time_col, status_col, "age", "gender", "stage_high", gene]
            work = merged[columns].dropna().copy()
            work = work[work[time_col] > 0]
            if work[gene].max() > 50:
                work[gene] = np.log2(work[gene].clip(lower=0) + 1)
            design = pd.DataFrame({
                "expression_z": zscore(work[gene], nan_policy="omit"),
                "age_z": zscore(work["age"], nan_policy="omit"),
                "gender": work["gender"].to_numpy(),
                "stage_high": work["stage_high"].to_numpy(),
            }, index=work.index)
            try:
                rows = ph_test(work[time_col], work[status_col], design, "TCGA-LUAD", endpoint, gene)
                details.extend(rows)
            except Exception as error:
                details.append({
                    "Cohort": "TCGA-LUAD", "Endpoint": endpoint, "Gene": gene,
                    "Covariate": "MODEL_FIT", "N": len(work), "Events": int(work[status_col].sum()),
                    "Time_Transform": "event-time rank/(events+1)", "Schoenfeld_R": np.nan,
                    "PH_P": np.nan, "Model_Status": f"failed: {type(error).__name__}",
                })
    detail = pd.DataFrame(details)
    detail["PH_FDR_within_Cohort_Endpoint_Covariate"] = detail.groupby(
        ["Cohort", "Endpoint", "Covariate"], group_keys=False
    )["PH_P"].transform(lambda x: fdr(x))
    return detail


def parse_gse_survival(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata.copy()
    if "Sample" in frame.columns:
        frame = frame.set_index("Sample")
    disease = frame.get("disease_state", pd.Series(index=frame.index, dtype=str)).astype(str).str.lower()
    frame = frame[disease.str.contains("adenocarcinoma", na=False)].copy()
    frame["event"] = frame["vital_status"].astype(str).str.lower().eq("dead").astype(int)
    frame["time_months"] = pd.to_numeric(frame["months_to_last_contact_or_death"], errors="coerce")
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame["male"] = frame["sex"].astype(str).str.lower().map({"male": 1.0, "female": 0.0})
    stage = frame["disease_stage"].astype(str).str.upper().str.replace(" ", "", regex=False)
    n_value = pd.to_numeric(stage.str.extract(r"PN([0-3])", expand=False), errors="coerce")
    frame["node_positive"] = np.where(n_value.notna(), n_value.ge(1).astype(float), np.nan)
    t_value = stage.str.extract(r"PT([1-4])", expand=False).astype(float)
    frame["t_high"] = np.where(t_value.notna(), t_value.ge(3).astype(float), np.nan)
    return frame[(frame["time_months"] > 0) & frame["time_months"].notna()]


def gse_ph() -> pd.DataFrame:
    metadata = parse_gse_survival(pd.read_csv(GSE_METADATA))
    expression = pd.read_csv(GSE_EXPRESSION, index_col=0)
    details = []
    for gene in expression.index.astype(str):
        work = metadata[["time_months", "event", "age", "male", "node_positive", "t_high"]].copy()
        work["expression"] = expression.loc[gene].reindex(work.index)
        work = work.dropna()
        if len(work) < 30 or work["event"].sum() < 10:
            continue
        design = pd.DataFrame({
            "expression_z": zscore(work["expression"]),
            "age_z": zscore(work["age"]),
            "male": work["male"].to_numpy(),
            "node_positive": work["node_positive"].to_numpy(),
            "t_high": work["t_high"].to_numpy(),
        }, index=work.index)
        try:
            rows = ph_test(work["time_months"], work["event"], design, "GSE68465", "overall_survival", gene)
            details.extend(rows)
        except Exception as error:
            details.append({
                "Cohort": "GSE68465", "Endpoint": "overall_survival", "Gene": gene,
                "Covariate": "MODEL_FIT", "N": len(work), "Events": int(work["event"].sum()),
                "Time_Transform": "event-time rank/(events+1)", "Schoenfeld_R": np.nan,
                "PH_P": np.nan, "Model_Status": f"failed: {type(error).__name__}",
            })
    detail = pd.DataFrame(details)
    detail["PH_FDR_within_Cohort_Endpoint_Covariate"] = detail.groupby(
        ["Cohort", "Endpoint", "Covariate"], group_keys=False
    )["PH_P"].transform(lambda x: fdr(x))
    return detail


def ph_model_summary(detail: pd.DataFrame) -> pd.DataFrame:
    valid = detail[detail["Covariate"] != "MODEL_FIT"].copy()
    rows = []
    for (cohort, endpoint, gene), part in valid.groupby(["Cohort", "Endpoint", "Gene"], sort=False):
        gene_part = part[part["Covariate"] == "expression_z"]
        rows.append({
            "Cohort": cohort,
            "Endpoint": endpoint,
            "Gene": gene,
            "N": int(part["N"].iloc[0]),
            "Events": int(part["Events"].iloc[0]),
            "Minimum_Covariate_PH_P": part["PH_P"].min(),
            "Minimum_Covariate_PH_FDR": part["PH_FDR_within_Cohort_Endpoint_Covariate"].min(),
            "Any_Covariate_Nominal_Violation": bool((part["PH_P"] < 0.05).any()),
            "Any_Covariate_FDR_Violation": bool((part["PH_FDR_within_Cohort_Endpoint_Covariate"] < 0.05).any()),
            "Expression_PH_P": gene_part["PH_P"].iloc[0] if len(gene_part) else np.nan,
            "Expression_PH_FDR": gene_part["PH_FDR_within_Cohort_Endpoint_Covariate"].iloc[0] if len(gene_part) else np.nan,
            "Expression_Nominal_Violation": bool((gene_part["PH_P"] < 0.05).any()),
            "Expression_FDR_Violation": bool((gene_part["PH_FDR_within_Cohort_Endpoint_Covariate"] < 0.05).any()),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    labels = pd.read_csv(NODE_LABELS).set_index("node_id")["clinical_target_label"].astype(int)
    oof = pd.read_csv(OOF)
    subset = oof[oof["method"].isin([PRIMARY, DEGREE])]
    pivots = {
        int(repeat): part.pivot(index="node_id", columns="method", values="score").sort_index()
        for repeat, part in subset.groupby("repeat")
    }
    data = torch.load(RAW_DATASET, map_location="cpu", weights_only=False)
    degree_all = np.bincount(data.edge_index_ppi[0].cpu().numpy(), minlength=data.num_nodes)
    common_nodes = sorted(set(labels.index).intersection(*[set(p.index) for p in pivots.values()]))
    degree = degree_all[np.asarray(common_nodes, dtype=int)]
    strata = pd.qcut(pd.Series(np.log1p(degree)), q=10, labels=False, duplicates="drop").fillna(0).astype(int).to_numpy()
    design = pd.DataFrame({"node_id": common_nodes, "label": labels.loc[common_nodes].to_numpy(), "stratum": strata})
    rng = np.random.default_rng(SEED)
    rows = []
    for replicate in range(1, BOOTSTRAPS + 1):
        sampled_nodes = []
        for stratum in sorted(design["stratum"].unique()):
            part = design[design["stratum"] == stratum]
            positive = part.loc[part["label"] == 1, "node_id"].to_numpy(dtype=int)
            unlabeled = part.loc[part["label"] == 0, "node_id"].to_numpy(dtype=int)
            if len(positive) and len(unlabeled):
                sampled_nodes.extend(rng.choice(positive, size=len(positive), replace=True).tolist())
                sampled_nodes.extend(rng.choice(unlabeled, size=len(positive), replace=True).tolist())
        sampled_nodes = np.asarray(sampled_nodes, dtype=int)
        repeat_primary, repeat_degree = [], []
        for repeat in sorted(pivots):
            sampled = pivots[repeat].loc[sampled_nodes]
            y = labels.loc[sampled_nodes].to_numpy(dtype=int)
            repeat_primary.append(average_precision_score(y, sampled[PRIMARY]))
            repeat_degree.append(average_precision_score(y, sampled[DEGREE]))
        ap_primary = float(np.mean(repeat_primary))
        ap_degree = float(np.mean(repeat_degree))
        rows.append({
            "Bootstrap_Replicate": replicate,
            "ctMV_GCN_AUPRC": ap_primary,
            "Degree_Logistic_AUPRC": ap_degree,
            "ctMV_GCN_minus_Degree": ap_primary - ap_degree,
        })
    boot = pd.DataFrame(rows)
    delta = boot["ctMV_GCN_minus_Degree"].to_numpy()
    lower_tail = (np.count_nonzero(delta <= 0) + 1) / (BOOTSTRAPS + 1)
    upper_tail = (np.count_nonzero(delta >= 0) + 1) / (BOOTSTRAPS + 1)
    summary = pd.DataFrame([{
        "Bootstrap_Unit": "gene cluster; the same resampled genes were evaluated in all three repeat-level OOF datasets",
        "Matching": "1:1 positive-unlabeled within each of ten empirical STRING-degree strata",
        "Bootstrap_Replicates": BOOTSTRAPS,
        "Random_Seed": SEED,
        "Gene_N": len(design),
        "Positive_Gene_N": int(design["label"].sum()),
        "Repeat_N": len(pivots),
        "ctMV_GCN_AUPRC_Median": boot["ctMV_GCN_AUPRC"].median(),
        "Degree_Logistic_AUPRC_Median": boot["Degree_Logistic_AUPRC"].median(),
        "Difference_Median": np.median(delta),
        "Difference_CI_Lower": np.quantile(delta, 0.025),
        "Difference_CI_Upper": np.quantile(delta, 0.975),
        "Two_Sided_Bootstrap_P": min(1.0, 2 * min(lower_tail, upper_tail)),
    }])
    return boot, summary


def configure_paths(args: argparse.Namespace) -> None:
    global REPO, RESULTS, OUT, TCGA_EXPRESSION, TCGA_COHORT, GSE_METADATA
    global GSE_EXPRESSION, OOF, NODE_LABELS, TOP20, RAW_DATASET
    REPO = args.repo_root.resolve()
    RESULTS = args.results_root.resolve() if args.results_root else REPO / "results"
    OUT = args.output_dir.resolve() if args.output_dir else RESULTS / "sensitivity"
    TCGA_EXPRESSION = args.tcga_expression.resolve()
    TCGA_COHORT = args.tcga_cohort.resolve() if args.tcga_cohort else RESULTS / "06_tcga" / "TCGA_primary_tumor_survival_cohort.csv"
    GSE_METADATA = args.gse_metadata.resolve()
    GSE_EXPRESSION = args.gse_expression.resolve()
    OOF = RESULTS / "02_benchmark" / "oof_predictions.csv.gz"
    NODE_LABELS = RESULTS / "01_data" / "node_label_audit.csv"
    TOP20 = RESULTS / "08_robustness" / "robust_top20.csv"
    RAW_DATASET = args.raw_dataset.resolve() if args.raw_dataset else RESULTS / "01_data" / "GNN_Dataset_Raw.pt"
    OUT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final Cox-PH diagnostics and gene-cluster bootstrap sensitivity analysis.")
    parser.add_argument("--repo-root", type=Path, default=REPO, help="Repository root; defaults to the repository containing this script.")
    parser.add_argument("--results-root", type=Path, help="Frozen v1.0.2 results directory; defaults to <repo-root>/results.")
    parser.add_argument("--tcga-expression", type=Path, required=True, help="UCSC Xena TCGA.LUAD.sampleMap_HiSeqV2/HiSeqV2 file.")
    parser.add_argument("--tcga-cohort", type=Path, help="TCGA_primary_tumor_survival_cohort.csv; defaults to the results directory.")
    parser.add_argument("--gse-metadata", type=Path, required=True, help="GSE68465_sample_metadata.csv.gz processed cache.")
    parser.add_argument("--gse-expression", type=Path, required=True, help="GSE68465_robust_top20_expression.csv.gz processed cache.")
    parser.add_argument("--raw-dataset", type=Path, help="GNN_Dataset_Raw.pt; defaults to the results directory.")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to <results-root>/sensitivity.")
    return parser.parse_args()


def main() -> None:
    configure_paths(parse_args())
    required = [TCGA_EXPRESSION, TCGA_COHORT, GSE_METADATA, GSE_EXPRESSION, OOF, NODE_LABELS, TOP20, RAW_DATASET]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    tcga_detail = tcga_ph()
    gse_detail = gse_ph()
    ph_detail = pd.concat([tcga_detail, gse_detail], ignore_index=True)
    ph_summary = ph_model_summary(ph_detail)
    ph_detail.to_csv(OUT / "cox_ph_schoenfeld_covariate_tests.csv", index=False, encoding="utf-8-sig")
    ph_summary.to_csv(OUT / "cox_ph_model_summary.csv", index=False, encoding="utf-8-sig")

    boot, boot_summary = cluster_bootstrap()
    boot.to_csv(OUT / "gene_cluster_bootstrap_replicates.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(OUT / "gene_cluster_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    interpretation = {
        "ph_test_method": "Pearson correlation of Schoenfeld residuals with ranked event time; BH FDR within each cohort/endpoint family",
        "ph_nominal_violations": int((ph_detail["PH_P"] < 0.05).sum()),
        "ph_fdr_violations": int((ph_detail["PH_FDR_within_Cohort_Endpoint_Covariate"] < 0.05).sum()),
        "models_with_any_covariate_nominal_violation": int(ph_summary["Any_Covariate_Nominal_Violation"].sum()),
        "models_with_any_covariate_fdr_violation": int(ph_summary["Any_Covariate_FDR_Violation"].sum()),
        "models_with_expression_nominal_violation": int(ph_summary["Expression_Nominal_Violation"].sum()),
        "models_with_expression_fdr_violation": int(ph_summary["Expression_FDR_Violation"].sum()),
        "cluster_bootstrap": boot_summary.iloc[0].to_dict(),
        "python": sys.version,
    }
    (OUT / "final_sensitivity_manifest.json").write_text(
        json.dumps(interpretation, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(json.dumps(interpretation, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
