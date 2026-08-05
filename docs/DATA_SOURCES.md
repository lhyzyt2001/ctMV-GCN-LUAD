# Data sources, frozen inputs and redistribution boundary

## Source datasets

| Resource | Version or accession | Role | Authoritative access |
|---|---|---|---|
| TISCH / GEO | GSE131907 | 15 aggregated cell-type expression features | [NCBI GEO GSE131907](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907); [TISCH](https://tisch.comp-genomics.org/) |
| STRING | v12.0, human taxon 9606 | high-confidence functional-association view, combined score >=700 | [STRING downloads](https://string-db.org/cgi/download?species_text=Homo+sapiens) |
| MSigDB / KEGG Medicus | `c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt` | weighted shared-pathway graph | [MSigDB human collections](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp) |
| Open Targets | LUAD `EFO_0000571`, export dated 2026-03-23 | clinical-stage target label | [Open Targets LUAD page](https://platform.opentargets.org/disease/EFO_0000571); [GraphQL API](https://platform.opentargets.org/api) |
| TCGA-LUAD | UCSC Xena HiSeqV2 and clinical matrix | paired expression, stage and outcome analyses | [UCSC Xena data pages](https://xenabrowser.net/datapages/) |
| GEO | GSE68465, platform GPL96 | independent expression and survival characterization | [NCBI GEO GSE68465](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68465); [GPL96](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96) |
| CPTAC LUAD | cBioPortal study `luad_cptac_2020` | tumor protein-profile characterization | [cBioPortal study](https://www.cbioportal.org/study/summary?id=luad_cptac_2020); [public API](https://www.cbioportal.org/api) |
| DepMap | Public 26Q1 | LUAD CRISPR Chronos gene effects | [current release](https://depmap.org/portal/data_page/?tab=currentRelease); [custom downloads](https://depmap.org/portal/data_page/?tab=customDownloads) |
| MSigDB | Hallmark and GO-BP snapshots | pre-ranked GSEA | versions and retrieved collection names are recorded in `results/05_interpretability/gene_set_manifest.json` |

## Exact input snapshots

The files used for the reported analysis, their byte counts and SHA-256 digests are recorded in [`INPUT_CHECKSUMS.tsv`](INPUT_CHECKSUMS.tsv). After downloading the licensed/public source files and arranging them as described in [`../data/README.md`](../data/README.md), run:

```bash
python src/run_all.py --check-inputs
```

The command exits with a non-zero status if a required file is absent or if a frozen input has the wrong digest. The checksums identify the exact study snapshots; they do not grant redistribution rights.

## TISCH-derived feature snapshot

`results/01_data/celltype_expression_features.csv` is the frozen 25,242-gene by 15-cell-type matrix used by the graph pipeline. It is byte-identical to the original `scRNA_multichannel_features_TISCH.csv` after accounting for the UTF-8 byte-order mark added to the repository copy. If `data/core/scRNA_multichannel_features_TISCH.csv` is absent, the code automatically uses this packaged snapshot. This removes an otherwise undocumented manual preprocessing dependency while preserving the GSE131907/TISCH provenance.

## Open Targets label snapshot

The original export was obtained for disease `EFO_0000571` on 2026-03-23 and frozen as `OT-EFO_0000571-associated-targets-2026_3_23-v0_0.tsv`. A gene is positive only when `maxClinicalTrialPhase > 0`; `globalScore` is not used as the endpoint. Because the Open Targets web export can change over time, the exact processed labels and full gene-level label audit are included as:

- `results/01_data/clinical_target_labels.csv`;
- `results/01_data/open_targets_label_audit.csv`.

If the original TSV is unavailable, `1.2_label.py` uses these frozen repository snapshots and records that fallback in `label_definition.json`.

## External-input details

- GSE68465 uses the GEO series matrix and the compressed GPL96 annotation file named `GPL96.annot.gz`.
- CPTAC protein values are requested from the cBioPortal public API using molecular profile and sample list `luad_cptac_2020_protein_quantification`; cached patient-level API responses are not redistributed.
- The reported DepMap analysis used a Public 26Q1 custom download restricted to the fixed robust top-20 genes and containing cell-line metadata. Users may alternatively supply the full `CRISPRGeneEffect.csv` and `Model.csv` release files.

## Why other raw data are not in this repository

The repository contains code and publication-level derived result tables needed to audit the manuscript. It does not redistribute raw STRING, MSigDB/KEGG, TCGA, CPTAC, GEO or DepMap downloads. Users must obtain those files from the original repositories and accept the applicable access or licensing terms.

Patient-level TCGA working tables, external input caches, trained model weights and intermediate `.pt`, `.pth`, `.npy` and `.joblib` files are excluded. Their absence does not remove the aggregate result tables, out-of-fold gene predictions or candidate-level evidence used in the manuscript.

## Endpoint provenance and claim boundaries

- The endpoint reflects existing clinical-stage evidence and may contain research, tractability and network-degree biases.
- GSE68465 contains 443 LUAD and 19 normal samples; survival models adjust for age, sex, nodal positivity and pT3-4.
- CPTAC values are tumor protein profiles normalized to a study-specific zero reference, not paired tumor-normal differences.
- DepMap cell-line dependency is supporting functional evidence only and does not establish patient efficacy.
- No external outcome is used to choose the robustness-aware top 20.
