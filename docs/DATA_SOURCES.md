# Data sources and redistribution boundary

## Source datasets

| Resource | Version or accession | Role | Access |
|---|---|---|---|
| TISCH / GEO | GSE131907 | 15 aggregated cell-type expression features | TISCH and NCBI GEO |
| STRING | v12.0, human taxon 9606 | high-confidence functional-association view | STRING downloads; combined score threshold 700 |
| MSigDB / KEGG Medicus | `c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt` | shared-pathway graph | MSigDB account and applicable terms |
| Open Targets | LUAD `EFO_0000571`, snapshot dated 2026-03-23 | clinical-stage target label | Open Targets Platform/downloads |
| TCGA-LUAD | UCSC Xena HiSeqV2 and clinical matrix | paired expression, stage and outcome analyses | UCSC Xena / NCI GDC |
| GEO | GSE68465, platform GPL96 | independent expression and survival characterization | NCBI GEO |
| CPTAC LUAD | `luad_cptac_2020` | tumor protein-profile characterization | cBioPortal public API |
| DepMap | Public 26Q1 | LUAD CRISPR Chronos gene effects | DepMap portal/custom downloads |
| MSigDB | Hallmark and GO-BP snapshots recorded in `results/05_interpretability/gene_set_manifest.json` | pre-ranked GSEA | MSigDB and applicable terms |

## Why raw data are not in this repository

The repository contains code and derived result tables needed to audit the manuscript. It does not redistribute raw STRING, MSigDB/KEGG, TCGA, CPTAC, GEO or DepMap downloads. Users must obtain those files from the original repositories and accept the applicable access or licensing terms.

Patient-level TCGA working tables, external input caches, trained model weights and intermediate `.pt`, `.pth`, `.npy` and `.joblib` files are also excluded. Their absence does not remove the aggregate result tables, out-of-fold gene predictions or candidate-level evidence used in the manuscript.

## Endpoint provenance

A graph gene is positive when the frozen Open Targets LUAD association table has `maxClinicalTrialPhase > 0`. The Open Targets global association score is not used as the label. This endpoint reflects existing clinical-stage evidence and may contain research, tractability and network-degree biases.

## External validation claim boundaries

- GSE68465 contains 443 LUAD and 19 normal samples; survival models adjust for age, sex, nodal positivity and pT3-4.
- CPTAC values are tumor protein profiles normalized to a study-specific zero reference, not paired tumor-normal differences.
- DepMap cell-line dependency is supporting functional evidence only and does not establish patient efficacy.
- No external outcome is used to choose the robustness-aware top 20.

