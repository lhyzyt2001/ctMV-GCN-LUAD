# Results index

This index maps the main manuscript claims to frozen repository outputs.

## Dataset and graph construction

- `results/01_data/dataset_qc.json`: node, label, feature and edge counts.
- `results/01_data/label_definition.json`: Open Targets endpoint definition.
- `results/01_data/ppi_qc.json`: STRING threshold and functional-association terminology.
- `results/01_data/celltype_similarity_qc.json`: mutual-KNN construction and aggregated-feature scope.
- `results/01_data/pathway_qc.json`: weighted shared-pathway construction.

## Predictive benchmark

- `results/02_benchmark/benchmark_design.json`: transductive label-separation rules.
- `results/02_benchmark/primary_benchmark_summary.csv`: primary comparator metrics.
- `results/02_benchmark/benchmark_summary.csv`: full benchmark including GNN ablations.
- `results/02_benchmark/fold_metrics.csv`: fold-level metrics.
- `results/02_benchmark/oof_predictions.csv.gz`: gene-level out-of-fold predictions.
- `results/02_benchmark/paired_bootstrap_auprc.csv`: repeat-aware AUPRC intervals and paired differences.

## Final ensemble and Top-K recovery

- `results/03_model/training_manifest.json`: architecture, loss, seed and epoch record.
- `results/03_model/all_gene_predictions.csv.gz`: ten-seed genome-wide predictions.
- `results/03_model/unlabeled_candidate_ranking.csv`: full unlabeled ranking.
- `results/03_model/attention_stability_summary.csv`: graph-view attention summary.
- `results/04_topk/topk_metrics_summary.csv`: mean Top-K recovery across repeats.

## Degree and topology robustness

- `results/07_validation/view_degree_bias_summary.csv`: raw score-degree correlations.
- `results/08_robustness/degree_matched_design.json`: matched-cohort prevalence and interpretation rule.
- `results/08_robustness/degree_matched_auprc_comparisons.csv`: paired matched AUPRC comparisons.
- `results/08_robustness/degree_conditional_permutation_test.csv`: conditional null test.
- `results/08_robustness/degree_residualized_oof_metrics.csv`: nonlinear residualization result.
- `results/08_robustness/edge_dropout_rank_stability.csv`: inference-time edge-removal sensitivity.

## Candidate selection and interpretation

- `results/08_robustness/robust_candidate_selection_manifest.json`: post hoc timing, filters and score formula.
- `results/08_robustness/robust_candidate_ranking.csv`: all eligible robustness scores.
- `results/08_robustness/robust_top20.csv`: fixed top-20 candidate set.
- `results/05_interpretability/candidate_channel_perturbation.csv`: channel-masking score changes.
- `results/05_interpretability/candidate_celltype_specificity_zscores.csv`: cell-type feature patterns.
- `results/05_interpretability/gsea_sensitivity_summary.csv`: primary, degree-residualized and structural-filter sensitivity counts.
- `results/05_interpretability/unlabeled_candidate_gsea_*.csv`: complete GSEA results.

## TCGA and independent computational characterization

- `results/07_validation/TCGA_paired_tumor_normal_robust_top20.csv`: 58-pair expression analysis.
- `results/07_validation/TCGA_stage_association_robust_top20.csv`: early- versus late-stage analysis.
- `results/06_tcga/TCGA_LUAD_robust_top20_survival_results.csv`: adjusted overall-survival models.
- `results/06_tcga/TCGA_LUAD_robust_top20_new_tumor_event_results.csv`: adjusted secondary endpoint.
- `results/08_robustness/external_validation/GSE68465_tumor_normal_validation.csv`: independent expression analysis.
- `results/08_robustness/external_validation/GSE68465_survival_validation.csv`: independent adjusted survival models.
- `results/08_robustness/external_validation/CPTAC_LUAD_protein_validation.csv`: tumor protein-profile results.
- `results/08_robustness/external_validation/depmap_luad_dependency_validation.csv`: LUAD CRISPR gene effects.
- `results/08_robustness/external_validation/external_validation_with_dependency_summary.csv`: integrated candidate-level summary.

## Figures

Publication figures are supplied as 300-dpi PNG and editable PDF where available. The workflow figure is also supplied as editable SVG. Manuscript composite panels are under `results/_composites`.

