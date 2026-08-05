from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import mannwhitneyu, rankdata, wilcoxon, zscore
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff
from statsmodels.stats.multitest import multipletests

from config import EXTERNAL_DATA_ROOT, ROBUSTNESS_DIR, ensure_result_dirs
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


EXTERNAL_DIR = ROBUSTNESS_DIR / "external_validation"
RAW_DIR = EXTERNAL_DATA_ROOT / "processed_cache"
GSE_MATRIX_FILE = EXTERNAL_DATA_ROOT / "GSE68465_series_matrix.txt.gz"
GPL_ANNOTATION_FILE = EXTERNAL_DATA_ROOT / "GPL96.annot.gz"
CBIOPORTAL_API = "https://www.cbioportal.org/api"
CPTAC_PROFILE = "luad_cptac_2020_protein_quantification"
CPTAC_SAMPLE_LIST = "luad_cptac_2020_protein_quantification"
BOOTSTRAPS = 2000
RANDOM_SEED = 42


def signed_rank_biserial(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return np.nan
    ranks = rankdata(np.abs(values))
    return float(np.sum(np.sign(values) * ranks) / np.sum(ranks))


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    medians = np.median(rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True), axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def add_fdr(frame: pd.DataFrame, p_column: str, output_column: str) -> pd.DataFrame:
    frame[output_column] = np.nan
    valid = frame[p_column].notna()
    if valid.any():
        frame.loc[valid, output_column] = multipletests(frame.loc[valid, p_column], method="fdr_bh")[1]
    return frame


def fetch_cptac_protein(genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    raw_cache = RAW_DIR / "CPTAC_LUAD_protein_top20_raw.csv.gz"
    mapping_cache = RAW_DIR / "CPTAC_LUAD_gene_mapping.csv"
    input_raw_cache = EXTERNAL_DATA_ROOT / "CPTAC_LUAD_protein_top20_raw.csv.gz"
    input_mapping_cache = EXTERNAL_DATA_ROOT / "CPTAC_LUAD_gene_mapping.csv"
    try:
        gene_response = requests.post(
            f"{CBIOPORTAL_API}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY",
            json=genes,
            headers=headers,
            timeout=60,
        )
        gene_response.raise_for_status()
        gene_records = gene_response.json()
        gene_map = {record["hugoGeneSymbol"]: int(record["entrezGeneId"]) for record in gene_records}
        reverse_map = {value: key for key, value in gene_map.items()}
        data_response = requests.post(
            f"{CBIOPORTAL_API}/molecular-profiles/{CPTAC_PROFILE}/molecular-data/fetch?projection=SUMMARY",
            json={"sampleListId": CPTAC_SAMPLE_LIST, "entrezGeneIds": list(reverse_map)},
            headers=headers,
            timeout=120,
        )
        data_response.raise_for_status()
        raw = pd.DataFrame(data_response.json())
        if raw.empty:
            raw = pd.DataFrame(columns=["sampleId", "entrezGeneId", "value"])
        raw["Gene"] = raw.get("entrezGeneId", pd.Series(dtype=float)).map(reverse_map)
        raw.to_csv(raw_cache, index=False, compression="gzip")
        pd.DataFrame(gene_records).to_csv(mapping_cache, index=False, encoding="utf-8-sig")
        source_status = "downloaded from cBioPortal public API"
    except requests.RequestException as error:
        if not raw_cache.exists() and input_raw_cache.exists():
            raw_cache = input_raw_cache
        if not mapping_cache.exists() and input_mapping_cache.exists():
            mapping_cache = input_mapping_cache
        if not raw_cache.exists() or not mapping_cache.exists():
            raise
        raw = pd.read_csv(raw_cache)
        gene_records = pd.read_csv(mapping_cache).to_dict("records")
        source_status = f"validated local cache used because API access failed: {type(error).__name__}"
    required_raw_columns = {"Gene", "value", "sampleId"}
    if not required_raw_columns.issubset(raw.columns):
        raise ValueError(f"CPTAC cache is missing columns: {sorted(required_raw_columns - set(raw.columns))}")
    cache_genes = set(raw["Gene"].dropna().astype(str))
    requested_returned = sorted(cache_genes.intersection(genes))
    (EXTERNAL_DIR / "CPTAC_data_source_status.json").write_text(json.dumps({
        "status": source_status,
        "requested_gene_count": len(genes),
        "genes_with_returned_protein_data": len(requested_returned),
        "profile": CPTAC_PROFILE,
        "sample_list": CPTAC_SAMPLE_LIST,
    }, indent=2), encoding="utf-8")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for rank, gene in enumerate(genes, start=1):
        values = pd.to_numeric(raw.loc[raw["Gene"] == gene, "value"], errors="coerce").dropna().to_numpy()
        if len(values):
            try:
                p_value = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            lower, upper = bootstrap_median_ci(values, rng)
            rows.append({
                "Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": "tested",
                "N_Protein_Profiles": len(values), "Median_CPTAC_Protein_Profile_Value": float(np.median(values)),
                "Median_CI_Lower": lower, "Median_CI_Upper": upper,
                "Signed_Rank_Biserial": signed_rank_biserial(values), "Wilcoxon_P_vs_Zero": p_value,
            })
        else:
            rows.append({
                "Robust_Candidate_Rank": rank, "Gene": gene,
                "Mapping_Status": "not returned by cBioPortal protein profile",
                "N_Protein_Profiles": 0, "Median_CPTAC_Protein_Profile_Value": np.nan,
                "Median_CI_Lower": np.nan, "Median_CI_Upper": np.nan,
                "Signed_Rank_Biserial": np.nan, "Wilcoxon_P_vs_Zero": np.nan,
            })
    result = add_fdr(pd.DataFrame(rows), "Wilcoxon_P_vs_Zero", "Wilcoxon_FDR")
    result.to_csv(EXTERNAL_DIR / "CPTAC_LUAD_protein_validation.csv", index=False, encoding="utf-8-sig")
    return result, raw


def read_gse_metadata(path: Path) -> pd.DataFrame:
    sample_ids: list[str] | None = None
    records: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            values = next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))
            key, entries = values[0], values[1:]
            if key == "!Sample_geo_accession":
                sample_ids = entries
            elif key == "!Sample_title":
                records["title"] = entries
            elif key == "!Sample_characteristics_ch1":
                parsed = [entry.split(":", 1) for entry in entries]
                names = [item[0].strip().lower() if len(item) == 2 else "unparsed" for item in parsed]
                most_common = pd.Series(names).mode().iloc[0]
                records[most_common] = [item[1].strip() if len(item) == 2 else item[0].strip() for item in parsed]
    if sample_ids is None:
        raise ValueError("GSE68465 sample accessions were not found")
    frame = pd.DataFrame(index=sample_ids)
    for key, values in records.items():
        if len(values) == len(frame):
            frame[key] = values
    frame.index.name = "Sample"
    return frame


def read_gpl_mapping(path: Path) -> pd.DataFrame:
    header_line = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("!platform_table_begin"):
                header_line = line_number + 1
                break
    if header_line is None:
        raise ValueError("GPL96 platform table marker was not found")
    annotation = pd.read_csv(
        path, sep="\t", comment="!", compression="gzip", dtype=str, skiprows=header_line
    )
    id_column = next(column for column in annotation.columns if column.strip().upper() in {"ID", "ID_REF"})
    symbol_column = next(column for column in annotation.columns if column.strip().lower() in {"gene symbol", "gene_symbol"})
    mapping = annotation[[id_column, symbol_column]].rename(columns={id_column: "Probe", symbol_column: "Gene_Raw"}).dropna()
    mapping["Gene"] = mapping["Gene_Raw"].astype(str).str.split(r"\s*///\s*|\s*//\s*|;", regex=True)
    mapping = mapping.explode("Gene")
    mapping["Gene"] = mapping["Gene"].astype(str).str.strip().str.upper()
    mapping = mapping[mapping["Gene"].str.match(r"^[A-Z0-9_.-]+$")]
    return mapping[["Probe", "Gene"]].drop_duplicates()


def read_gse_expression(path: Path) -> pd.DataFrame:
    expression = pd.read_csv(path, sep="\t", comment="!", compression="gzip", quotechar='"', index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str).str.strip('"')
    expression = expression.apply(pd.to_numeric, errors="coerce")
    if np.nanpercentile(expression.to_numpy(dtype=float), 99) > 50:
        expression = np.log2(expression.clip(lower=0) + 1)
    return expression


def select_gene_expression(
    expression: pd.DataFrame, mapping: pd.DataFrame, genes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, selected = [], {}
    for rank, gene in enumerate(genes, start=1):
        probes = mapping.loc[mapping["Gene"] == gene, "Probe"].drop_duplicates()
        probes = [probe for probe in probes if probe in expression.index]
        if probes:
            iqr = expression.loc[probes].quantile(0.75, axis=1) - expression.loc[probes].quantile(0.25, axis=1)
            probe = str(iqr.idxmax())
            selected[gene] = expression.loc[probe]
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Selected_Probe": probe, "Available_Probe_Count": len(probes), "Selection_Rule": "maximum IQR; outcome-blind"})
        else:
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Selected_Probe": np.nan, "Available_Probe_Count": 0, "Selection_Rule": "not mapped"})
    return pd.DataFrame(selected).T, pd.DataFrame(rows)


def gse_tumor_normal(
    gene_expression: pd.DataFrame, metadata: pd.DataFrame, genes: list[str]
) -> pd.DataFrame:
    disease = metadata.get("disease_state", pd.Series(index=metadata.index, dtype=str)).astype(str).str.lower()
    tumor_samples = metadata.index[disease.str.contains("adenocarcinoma", na=False)]
    normal_samples = metadata.index[disease.str.contains("normal", na=False)]
    rows = []
    for rank, gene in enumerate(genes, start=1):
        if gene not in gene_expression.index:
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": "not mapped", "Tumor_N": 0, "Normal_N": 0, "Tumor_Minus_Normal_Median": np.nan, "Cliffs_Delta": np.nan, "Mann_Whitney_P": np.nan})
            continue
        tumor = gene_expression.loc[gene, gene_expression.columns.intersection(tumor_samples)].dropna().to_numpy(dtype=float)
        normal = gene_expression.loc[gene, gene_expression.columns.intersection(normal_samples)].dropna().to_numpy(dtype=float)
        if len(tumor) and len(normal):
            test = mannwhitneyu(tumor, normal, alternative="two-sided")
            effect = 2 * test.statistic / (len(tumor) * len(normal)) - 1
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": "tested", "Tumor_N": len(tumor), "Normal_N": len(normal), "Tumor_Minus_Normal_Median": float(np.median(tumor) - np.median(normal)), "Cliffs_Delta": float(effect), "Mann_Whitney_P": float(test.pvalue)})
        else:
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": "group unavailable", "Tumor_N": len(tumor), "Normal_N": len(normal), "Tumor_Minus_Normal_Median": np.nan, "Cliffs_Delta": np.nan, "Mann_Whitney_P": np.nan})
    return add_fdr(pd.DataFrame(rows), "Mann_Whitney_P", "Mann_Whitney_FDR")


def parse_tnm(value: str) -> tuple[float, float]:
    text = str(value).upper().replace(" ", "")
    n_match = re.search(r"PN([0-3])", text)
    t_match = re.search(r"PT([1-4])", text)
    node_positive = float(int(n_match.group(1)) >= 1) if n_match else np.nan
    t_high = float(int(t_match.group(1)) >= 3) if t_match else np.nan
    return node_positive, t_high


def gse_survival(gene_expression: pd.DataFrame, metadata: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    frame = metadata.copy()
    disease = frame.get("disease_state", pd.Series(index=frame.index, dtype=str)).astype(str).str.lower()
    frame = frame[disease.str.contains("adenocarcinoma", na=False)].copy()
    frame["event"] = frame.get("vital_status", "").astype(str).str.lower().eq("dead").astype(int)
    frame["time_months"] = pd.to_numeric(frame.get("months_to_last_contact_or_death"), errors="coerce")
    frame["age"] = pd.to_numeric(frame.get("age"), errors="coerce")
    frame["male"] = frame.get("sex", "").astype(str).str.lower().map({"male": 1.0, "female": 0.0})
    tnm = frame.get("disease_stage", pd.Series(index=frame.index, dtype=str)).map(parse_tnm)
    frame["node_positive"] = [value[0] for value in tnm]
    frame["t_high"] = [value[1] for value in tnm]
    frame = frame[(frame["time_months"] > 0) & frame["time_months"].notna()]
    rows = []
    for rank, gene in enumerate(genes, start=1):
        if gene not in gene_expression.index:
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": "not mapped", "N": 0, "Events": 0, "Adjusted_HR_per_SD": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan, "Adjusted_Cox_P": np.nan, "LogRank_P": np.nan})
            continue
        work = frame[["time_months", "event", "age", "male", "node_positive", "t_high"]].copy()
        work["expression"] = gene_expression.loc[gene].reindex(work.index)
        work = work.dropna()
        if len(work) < 30 or work["event"].sum() < 10:
            rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": "insufficient complete cases", "N": len(work), "Events": int(work["event"].sum()), "Adjusted_HR_per_SD": np.nan, "CI_Lower": np.nan, "CI_Upper": np.nan, "Adjusted_Cox_P": np.nan, "LogRank_P": np.nan})
            continue
        high = (work["expression"] >= work["expression"].median()).astype(int)
        _, logrank_p = survdiff(work["time_months"], work["event"], high)
        design = pd.DataFrame({
            "expression_z": zscore(work["expression"]), "age_z": zscore(work["age"]),
            "male": work["male"].to_numpy(), "node_positive": work["node_positive"].to_numpy(),
            "t_high": work["t_high"].to_numpy(),
        }, index=work.index)
        try:
            fit = PHReg(work["time_months"], design, status=work["event"], ties="efron").fit(disp=0)
            gene_index = design.columns.get_loc("expression_z")
            hr = float(np.exp(fit.params[gene_index]))
            ci = np.exp(fit.conf_int()[gene_index])
            cox_p = float(fit.pvalues[gene_index])
            status = "tested"
        except Exception as error:
            hr, ci, cox_p, status = np.nan, (np.nan, np.nan), np.nan, f"model failed: {type(error).__name__}"
        rows.append({"Robust_Candidate_Rank": rank, "Gene": gene, "Mapping_Status": status, "N": len(work), "Events": int(work["event"].sum()), "Adjusted_HR_per_SD": hr, "CI_Lower": float(ci[0]), "CI_Upper": float(ci[1]), "Adjusted_Cox_P": cox_p, "LogRank_P": float(logrank_p)})
    result = add_fdr(pd.DataFrame(rows), "Adjusted_Cox_P", "Adjusted_Cox_FDR")
    return add_fdr(result, "LogRank_P", "LogRank_FDR")


def external_figures(cptac: pd.DataFrame, expression: pd.DataFrame, survival: pd.DataFrame) -> None:
    apply_bmc_style()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 4.1))

    protein = cptac.dropna(subset=["Median_CPTAC_Protein_Profile_Value"]).sort_values(
        "Median_CPTAC_Protein_Profile_Value"
    )
    for y, row in enumerate(protein.itertuples(index=False)):
        significant = pd.notna(row.Wilcoxon_FDR) and row.Wilcoxon_FDR < 0.05
        color = "#0072B2" if significant else "0.65"
        lower = max(0.0, row.Median_CPTAC_Protein_Profile_Value - row.Median_CI_Lower)
        upper = max(0.0, row.Median_CI_Upper - row.Median_CPTAC_Protein_Profile_Value)
        axes[0].errorbar(
            row.Median_CPTAC_Protein_Profile_Value, y,
            xerr=np.array([[lower], [upper]]), fmt="o", markersize=3,
            color=color, capsize=1.5,
        )
    axes[0].axvline(0, color="0.5", linestyle="--")
    axes[0].set(
        yticks=np.arange(len(protein)), yticklabels=protein["Gene"],
        xlabel="Median tumor protein profile value\n(study-specific zero reference)",
        title=f"a  CPTAC tumor protein profile\n{(protein['Wilcoxon_FDR'] < 0.05).sum()}/{len(protein)} FDR<0.05",
    )

    tumor_normal = expression.dropna(subset=["Tumor_Minus_Normal_Median"]).sort_values(
        "Tumor_Minus_Normal_Median"
    )
    colors = np.where(tumor_normal["Mann_Whitney_FDR"] < 0.05, "#D55E00", "0.65")
    axes[1].scatter(
        tumor_normal["Tumor_Minus_Normal_Median"], np.arange(len(tumor_normal)),
        color=colors, s=13,
    )
    axes[1].axvline(0, color="0.5", linestyle="--")
    axes[1].set(
        yticks=np.arange(len(tumor_normal)), yticklabels=tumor_normal["Gene"],
        xlabel="Tumor - normal median expression",
        title=f"b  GSE68465 expression\n{(tumor_normal['Mann_Whitney_FDR'] < 0.05).sum()}/{len(tumor_normal)} FDR<0.05",
    )

    outcome = survival.dropna(subset=["Adjusted_HR_per_SD"]).sort_values("Adjusted_HR_per_SD")
    for y, row in enumerate(outcome.itertuples(index=False)):
        significant = pd.notna(row.Adjusted_Cox_FDR) and row.Adjusted_Cox_FDR < 0.05
        color = "#009E73" if significant else "0.65"
        lower = max(0.0, row.Adjusted_HR_per_SD - row.CI_Lower)
        upper = max(0.0, row.CI_Upper - row.Adjusted_HR_per_SD)
        axes[2].errorbar(
            row.Adjusted_HR_per_SD, y, xerr=np.array([[lower], [upper]]),
            fmt="o", markersize=3, color=color, capsize=1.5,
        )
    axes[2].axvline(1, color="0.5", linestyle="--")
    axes[2].set_xscale("log")
    axes[2].set(
        yticks=np.arange(len(outcome)), yticklabels=outcome["Gene"],
        xlabel="Adjusted HR per expression SD",
        title=f"c  GSE68465 survival\n{(outcome['Adjusted_Cox_FDR'] < 0.05).sum()}/{len(outcome)} FDR<0.05",
    )

    for ax in axes:
        ax.grid(axis="x", color="0.9", linewidth=0.5)
    fig.text(0.5, 0.015, "Colored points: FDR < 0.05; grey points: not significant. Error bars show 95% confidence intervals where available.", ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, EXTERNAL_DIR / "independent_external_validation")

def main() -> None:
    ensure_result_dirs()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    top20 = pd.read_csv(ROBUSTNESS_DIR / "robust_top20.csv")
    genes = top20.sort_values("Robust_Candidate_Rank")["Gene_Symbol"].astype(str).tolist()
    cptac, _ = fetch_cptac_protein(genes)

    if not GSE_MATRIX_FILE.exists() or not GPL_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"GSE68465 series matrix and GPL96 annotation are required in {EXTERNAL_DATA_ROOT}")
    metadata = read_gse_metadata(GSE_MATRIX_FILE)
    expression = read_gse_expression(GSE_MATRIX_FILE)
    mapping = read_gpl_mapping(GPL_ANNOTATION_FILE)
    gene_expression, probe_audit = select_gene_expression(expression, mapping, genes)
    metadata.to_csv(RAW_DIR / "GSE68465_sample_metadata.csv.gz", compression="gzip")
    probe_audit.to_csv(EXTERNAL_DIR / "GSE68465_probe_selection_audit.csv", index=False, encoding="utf-8-sig")
    gene_expression.to_csv(RAW_DIR / "GSE68465_robust_top20_expression.csv.gz", compression="gzip")
    tumor_normal = gse_tumor_normal(gene_expression, metadata, genes)
    survival = gse_survival(gene_expression, metadata, genes)
    tumor_normal.to_csv(EXTERNAL_DIR / "GSE68465_tumor_normal_validation.csv", index=False, encoding="utf-8-sig")
    survival.to_csv(EXTERNAL_DIR / "GSE68465_survival_validation.csv", index=False, encoding="utf-8-sig")

    summary = top20[["Robust_Candidate_Rank", "Gene_Symbol", "Robustness_Score"]].rename(columns={"Gene_Symbol": "Gene"})
    summary = summary.merge(cptac[["Gene", "N_Protein_Profiles", "Median_CPTAC_Protein_Profile_Value", "Wilcoxon_FDR"]], on="Gene", how="left")
    summary = summary.merge(tumor_normal[["Gene", "Tumor_Minus_Normal_Median", "Cliffs_Delta", "Mann_Whitney_FDR"]], on="Gene", how="left")
    summary = summary.merge(survival[["Gene", "Adjusted_HR_per_SD", "Adjusted_Cox_FDR"]], on="Gene", how="left")
    summary["CPTAC_Profile_FDR_lt_0.05"] = summary["Wilcoxon_FDR"] < 0.05
    summary["GSE_TumorNormal_FDR_lt_0.05"] = summary["Mann_Whitney_FDR"] < 0.05
    summary["GSE_Survival_FDR_lt_0.05"] = summary["Adjusted_Cox_FDR"] < 0.05
    summary["CPTAC_Profile_Direction"] = np.select(
        [
            summary["CPTAC_Profile_FDR_lt_0.05"] & (summary["Median_CPTAC_Protein_Profile_Value"] > 0),
            summary["CPTAC_Profile_FDR_lt_0.05"] & (summary["Median_CPTAC_Protein_Profile_Value"] < 0),
        ],
        ["above study zero reference", "below study zero reference"],
        default="not significant or unavailable",
    )
    summary["GSE_Expression_Direction"] = np.select(
        [
            summary["GSE_TumorNormal_FDR_lt_0.05"] & (summary["Tumor_Minus_Normal_Median"] > 0),
            summary["GSE_TumorNormal_FDR_lt_0.05"] & (summary["Tumor_Minus_Normal_Median"] < 0),
        ],
        ["higher in tumor", "lower in tumor"],
        default="not significant or unavailable",
    )
    both_omic = summary["CPTAC_Profile_FDR_lt_0.05"] & summary["GSE_TumorNormal_FDR_lt_0.05"]
    same_sign = both_omic & (
        np.sign(summary["Median_CPTAC_Protein_Profile_Value"])
        == np.sign(summary["Tumor_Minus_Normal_Median"])
    )
    summary["Cross_Omic_Sign_Status"] = np.select(
        [same_sign, both_omic & ~same_sign, summary["CPTAC_Profile_FDR_lt_0.05"] | summary["GSE_TumorNormal_FDR_lt_0.05"]],
        ["same sign across platform-specific references", "opposite sign across platform-specific references", "one platform significant"],
        default="neither platform significant or unavailable",
    )
    summary["Concordant_Cross_Omic_Support"] = same_sign
    summary["Significant_Endpoint_Count"] = summary[["CPTAC_Profile_FDR_lt_0.05", "GSE_TumorNormal_FDR_lt_0.05", "GSE_Survival_FDR_lt_0.05"]].sum(axis=1)
    summary.to_csv(EXTERNAL_DIR / "external_validation_summary.csv", index=False, encoding="utf-8-sig")
    external_figures(cptac, tumor_normal, survival)
    manifest = {
        "candidate_selection": "Post hoc robustness-aware top 20 developed after degree-bias diagnosis; not preregistered or prespecified.",
        "CPTAC": {
            "study": "luad_cptac_2020", "source": "cBioPortal public API",
            "profile": CPTAC_PROFILE,
            "test": "two-sided one-sample Wilcoxon against the profile's study-specific zero reference",
            "claim_boundary": "The cBioPortal tumor-only profile is not a paired tumor-normal comparison and cannot establish cancer-specific protein differential abundance.",
        },
        "GSE68465": {
            "source": "NCBI GEO series matrix, GPL96", "probe_selection": "maximum IQR without outcome labels",
            "tumor_normal_test": "two-sided Mann-Whitney; 443 LUAD and 19 normal samples expected",
            "survival_model": "Cox PH per expression SD, adjusted for age, sex, nodal positivity, and pT3-4",
        },
        "multiple_testing": "Benjamini-Hochberg within each endpoint over all mapped robustness-aware candidates.",
        "direction_rule": "Cross-omic sign is reported explicitly. Because the CPTAC zero reference is not adjacent normal tissue, same sign is descriptive and is not labelled tumor-normal concordance.",
        "interpretation": "Expression, survival, and tumor protein profile values are separate supporting endpoints; significance is not counted as concordant evidence when signs oppose, and none demonstrates therapeutic efficacy or dependency.",
    }
    (EXTERNAL_DIR / "external_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
