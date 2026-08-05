from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from config import (
    DEPMAP_CUSTOM_FILE,
    DEPMAP_GENE_EFFECT_FILE,
    DEPMAP_MODEL_FILE,
    ROBUSTNESS_DIR,
    ensure_result_dirs,
)
from plot_style import FULL_WIDTH, apply_bmc_style, save_figure


EXTERNAL_DIR = ROBUSTNESS_DIR / "external_validation"
STATUS_FILE = EXTERNAL_DIR / "depmap_validation_status.json"
GENE_EFFECT_THRESHOLD = -0.5
FRACTION_THRESHOLD = 0.25


def add_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    frame["Wilcoxon_FDR_Less_Than_Zero"] = np.nan
    valid = frame["Wilcoxon_P_Less_Than_Zero"].notna()
    if valid.any():
        frame.loc[valid, "Wilcoxon_FDR_Less_Than_Zero"] = multipletests(
            frame.loc[valid, "Wilcoxon_P_Less_Than_Zero"], method="fdr_bh"
        )[1]
    return frame


def gene_symbol(column: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", str(column)).strip().upper()


def luad_model_ids(model: pd.DataFrame) -> list[str]:
    id_column = next((c for c in ("ModelID", "DepMap_ID", "depmap_id") if c in model), model.columns[0])
    text_columns = [
        c for c in (
            "OncotreeCode", "OncotreePrimaryDisease", "OncotreeSubtype",
            "DepmapModelType", "lineage_subtype", "primary_disease",
        ) if c in model
    ]
    if not text_columns:
        raise ValueError("DepMap model metadata lacks disease/context columns")
    text = model[text_columns].fillna("").astype(str).agg(" | ".join, axis=1)
    keep = text.str.contains(r"(?i)\bLUAD\b|lung adenocarcinoma")
    ids = model.loc[keep, id_column].astype(str).drop_duplicates().tolist()
    if not ids:
        raise ValueError("No LUAD models were identified in DepMap metadata")
    return ids


def main() -> None:
    ensure_result_dirs()
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    top20_file = ROBUSTNESS_DIR / "robust_top20.csv"
    has_custom = DEPMAP_CUSTOM_FILE.exists()
    has_full_release = DEPMAP_GENE_EFFECT_FILE.exists() and DEPMAP_MODEL_FILE.exists()
    missing = []
    if not top20_file.exists():
        missing.append(str(top20_file))
    if not has_custom and not has_full_release:
        missing.extend([
            str(DEPMAP_CUSTOM_FILE),
            f"or both {DEPMAP_GENE_EFFECT_FILE} and {DEPMAP_MODEL_FILE}",
        ])
    if missing:
        status = {
            "status": "not_run_missing_official_input",
            "missing_files": missing,
            "required_release": "DepMap Public 26Q1",
            "required_files": [
                "Portal custom CRISPR download with cell-line metadata",
                "or full CRISPRGeneEffect.csv plus Model.csv",
            ],
            "data_source": "https://depmap.org/portal/data_page/?tab=currentRelease",
            "interpretation": "No dependency result is claimed until real DepMap files are present.",
        }
        STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
        pd.DataFrame(columns=[
            "Robust_Candidate_Rank", "Gene", "Mapping_Status", "LUAD_Model_N",
            "Median_Gene_Effect", "IQR_Lower", "IQR_Upper",
            "Fraction_Gene_Effect_le_-0.5", "Fraction_Gene_Effect_le_-1.0",
            "Wilcoxon_P_Less_Than_Zero", "Wilcoxon_FDR_Less_Than_Zero",
            "Functional_Dependency_Support",
        ]).to_csv(EXTERNAL_DIR / "depmap_luad_dependency_validation.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(status, indent=2))
        return

    top20 = pd.read_csv(top20_file).sort_values("Robust_Candidate_Rank")
    genes = top20["Gene_Symbol"].astype(str).str.upper().tolist()
    input_snapshot = None
    if has_custom:
        input_snapshot = DEPMAP_CUSTOM_FILE.resolve()
        custom = pd.read_csv(DEPMAP_CUSTOM_FILE, low_memory=False)
        model_id_column = next(
            (column for column in ("depmap_id", "ModelID", "DepMap_ID") if column in custom),
            custom.columns[0],
        )
        context_columns = [
            column for column in custom
            if column.lower().startswith("lineage_")
            or column in {"OncotreeCode", "OncotreePrimaryDisease", "OncotreeSubtype", "DepmapModelType"}
        ]
        if not context_columns:
            raise ValueError("DepMap custom file must include cell-line metadata")
        context = custom[context_columns].fillna("").astype(str).agg(" | ".join, axis=1)
        custom = custom.loc[context.str.contains("lung adenocarcinoma", case=False, regex=False)].copy()
        custom = custom.set_index(model_id_column)
        gene_columns = {gene_symbol(column): column for column in custom.columns if column not in context_columns}
        effect = custom[[gene_columns[gene] for gene in genes if gene in gene_columns]].copy()
        input_mode = "portal custom download with cell-line metadata"
    else:
        model = pd.read_csv(DEPMAP_MODEL_FILE, low_memory=False)
        luad_ids = luad_model_ids(model)
        header = pd.read_csv(DEPMAP_GENE_EFFECT_FILE, nrows=0)
        model_id_column = header.columns[0]
        gene_columns = {gene_symbol(column): column for column in header.columns[1:]}
        selected_columns = [gene_columns[gene] for gene in genes if gene in gene_columns]
        effect = pd.read_csv(
            DEPMAP_GENE_EFFECT_FILE,
            usecols=[model_id_column, *selected_columns],
            low_memory=False,
        ).set_index(model_id_column)
        effect.index = effect.index.astype(str)
        effect = effect.loc[effect.index.intersection(luad_ids)]
        input_mode = "full release files"
    if effect.empty:
        raise ValueError("No LUAD cell lines with CRISPR data were identified")

    rows = []
    for rank, gene in enumerate(genes, start=1):
        column = gene_columns.get(gene)
        values = (
            pd.to_numeric(effect[column], errors="coerce").dropna().to_numpy(dtype=float)
            if column in effect else np.array([], dtype=float)
        )
        if len(values):
            try:
                p_value = float(wilcoxon(values, alternative="less", zero_method="wilcox").pvalue)
            except ValueError:
                p_value = 1.0
            rows.append({
                "Robust_Candidate_Rank": rank,
                "Gene": gene,
                "Mapping_Status": "tested",
                "LUAD_Model_N": len(values),
                "Median_Gene_Effect": float(np.median(values)),
                "IQR_Lower": float(np.quantile(values, 0.25)),
                "IQR_Upper": float(np.quantile(values, 0.75)),
                "Fraction_Gene_Effect_le_-0.5": float(np.mean(values <= GENE_EFFECT_THRESHOLD)),
                "Fraction_Gene_Effect_le_-1.0": float(np.mean(values <= -1.0)),
                "Wilcoxon_P_Less_Than_Zero": p_value,
            })
        else:
            rows.append({
                "Robust_Candidate_Rank": rank, "Gene": gene,
                "Mapping_Status": "gene or LUAD model unavailable", "LUAD_Model_N": 0,
                "Median_Gene_Effect": np.nan, "IQR_Lower": np.nan, "IQR_Upper": np.nan,
                "Fraction_Gene_Effect_le_-0.5": np.nan,
                "Fraction_Gene_Effect_le_-1.0": np.nan,
                "Wilcoxon_P_Less_Than_Zero": np.nan,
            })
    result = add_fdr(pd.DataFrame(rows))
    result["Functional_Dependency_Support"] = (
        (result["Wilcoxon_FDR_Less_Than_Zero"] < 0.05)
        & (
            (result["Median_Gene_Effect"] <= GENE_EFFECT_THRESHOLD)
            | (result["Fraction_Gene_Effect_le_-0.5"] >= FRACTION_THRESHOLD)
        )
    )
    result.to_csv(EXTERNAL_DIR / "depmap_luad_dependency_validation.csv", index=False, encoding="utf-8-sig")
    external_summary_file = EXTERNAL_DIR / "external_validation_summary.csv"
    if external_summary_file.exists():
        external_summary = pd.read_csv(external_summary_file)
        dependency_columns = result[[
            "Gene", "LUAD_Model_N", "Median_Gene_Effect",
            "Fraction_Gene_Effect_le_-0.5", "Wilcoxon_FDR_Less_Than_Zero",
            "Functional_Dependency_Support",
        ]]
        external_summary.merge(
            dependency_columns, on="Gene", how="left", validate="one_to_one"
        ).to_csv(
            EXTERNAL_DIR / "external_validation_with_dependency_summary.csv",
            index=False, encoding="utf-8-sig",
        )

    apply_bmc_style()
    plot = result.dropna(subset=["Median_Gene_Effect"]).sort_values("Median_Gene_Effect", ascending=False)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.72, 4.3))
    color = np.where(plot["Functional_Dependency_Support"], "#0072B2", "0.65")
    xerr = np.vstack([
        plot["Median_Gene_Effect"] - plot["IQR_Lower"],
        plot["IQR_Upper"] - plot["Median_Gene_Effect"],
    ])
    ax.errorbar(plot["Median_Gene_Effect"], np.arange(len(plot)), xerr=xerr, fmt="none", color="0.55", capsize=1.5)
    ax.scatter(plot["Median_Gene_Effect"], np.arange(len(plot)), c=color, s=18, zorder=3)
    ax.axvline(0, color="black", linewidth=0.7, label="No effect")
    ax.axvline(GENE_EFFECT_THRESHOLD, color="#D55E00", linestyle="--", linewidth=0.8, label="Dependency threshold (-0.5)")
    ax.set(
        yticks=np.arange(len(plot)), yticklabels=plot["Gene"],
        xlabel="DepMap Chronos gene effect (median and IQR)",
        ylabel="", title="LUAD cell-line CRISPR dependency",
    )
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    save_figure(fig, EXTERNAL_DIR / "depmap_luad_dependency")

    status = {
        "status": "completed",
        "release": "DepMap Public 26Q1",
        "input_mode": input_mode,
        "input_snapshot": f"external_data_root/{input_snapshot.name}" if input_snapshot else None,
        "data_source": "https://depmap.org/portal/data_page/?tab=customDownloads",
        "retrieval_date": "2026-07-21",
        "luad_model_count": int(effect.shape[0]),
        "tested_gene_count": int((result["Mapping_Status"] == "tested").sum()),
        "supported_gene_count": int(result["Functional_Dependency_Support"].sum()),
        "support_rule": "FDR<0.05 and median gene effect<=-0.5 or >=25% LUAD models with gene effect<=-0.5",
        "interpretation": "Cell-line dependency supports functional relevance but does not establish patient efficacy.",
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
