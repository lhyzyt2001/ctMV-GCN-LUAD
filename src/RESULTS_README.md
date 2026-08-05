# Final LUAD multiview-GNN results

## Evaluation and claim boundary

- The endpoint is Open Targets clinical evidence (`maxClinicalTrialPhase > 0`), with 1,184 positives among 25,242 graph genes (4.69%).
- Evaluation is repeated 5-fold outer cross-validation with three repeats. Outer-test labels are excluded from fitting, early stopping, threshold selection and RWR seed construction.
- All nodes and disease-independent graph topology remain visible during message passing. This is transductive node classification, not validation on an unseen graph or cohort.
- The robustness-aware top 20 was selected post hoc after diagnosing degree bias. It is neither preregistered nor prespecified, and external outcomes were not used for selection.

## Predictive results

- RWR remained the strongest benchmark (ROC-AUC 0.933 +/- 0.006; AUPRC 0.439 +/- 0.032).
- The network-degree logistic baseline also exceeded the GNN (ROC-AUC 0.888 +/- 0.008; AUPRC 0.318 +/- 0.021).
- The best GNN was local multiview attention with weighted BCE (ROC-AUC 0.851 +/- 0.015; AUPRC 0.215 +/- 0.033).
- Therefore, the manuscript must not claim that the GNN is the best overall predictor. The complete benchmark, including RWR and degree, should be reported; the GNN-only ablation figure may be used to explain the architecture.
- Because all methods are evaluated on the assembled graph, these numbers support within-graph prioritization only.

## Degree-bias robustness

- In 1:1 positive-negative matching within empirical STRING-degree strata, prevalence and random AUPRC are both 0.5. Matched AUPRC must not be compared numerically with the original 4.69%-prevalence benchmark.
- Within the same matched bootstrap samples, GNN AUPRC was 0.649 (95% interval 0.633-0.664), degree-only AUPRC was 0.627 (0.616-0.637), and the paired GNN-minus-degree difference was 0.0218 (0.0058-0.0380; P=0.0100).
- RWR remained stronger in the matched analysis: AUPRC 0.740 (0.728-0.752); GNN-minus-RWR was -0.0920 (-0.1099 to -0.0738; P=0.0010).
- The observed degree-stratified GNN ROC-AUC was 0.711 versus a conditional-permutation null mean of 0.501 (P=0.0005).
- Nonlinear residualization against six degree features reduced performance to ROC-AUC 0.680 and AUPRC 0.079. This remains above the original prevalence, but shows that topology explains substantial signal.
- Inference-time removal of up to 20% of edges is reported as rank-sensitivity analysis, not as a retrained performance benchmark.

## Unified robustness-aware candidates

The same post hoc top 20 is used in TCGA, external expression/protein and DepMap analyses:

CCDC80, LUM, CTSG, MFAP2, FMOD, CFP, PALMD, MASP1, CST2, ACTA2, TREML1, ISLR, APOA2, FCER1A, POSTN, AMBP, C4A, FCN3, PLAC9 and LTC4S.

- In 58 paired TCGA-LUAD tumor/adjacent-normal samples, 16 of 20 genes had FDR < 0.05. Effect estimates and confidence intervals must be reported with P values; APOA2 has a zero median paired difference despite a significant signed-rank result.
- Four genes had early-versus-late stage differences at FDR < 0.05: CTSG, TREML1, C4A and LTC4S.
- Overall-survival median-split log-rank FDR was below 0.05 for CTSG, TREML1 and LTC4S, but no continuous-expression, covariate-adjusted Cox association remained below FDR 0.05. These are exploratory TCGA associations, not independent prognostic validation.
- For the secondary TCGA new-tumor-event endpoint, TREML1 and LTC4S retained adjusted Cox FDR < 0.05. This is still internal evidence from TCGA and should be described as secondary.

## Independent computational validation

- GSE68465 contained 443 LUAD and 19 normal samples. Seventeen candidates mapped to GPL96 and 15 had tumor-normal expression differences at FDR < 0.05.
- None of the 17 mapped candidates had an adjusted survival association at FDR < 0.05 in GSE68465.
- CPTAC returned tumor protein profiles for 16 candidates; 14 differed from the study-specific zero reference at FDR < 0.05. This is not a paired tumor-normal protein comparison and cannot establish cancer-specific protein differential abundance.
- CTSG, APOA2 and AMBP had the same sign across the GSE tumor-normal contrast and the CPTAC study-specific zero reference. This is descriptive cross-platform sign agreement, not tumor-normal protein concordance. Several other genes had opposite signs.
- In DepMap Public 26Q1, 19 candidates were evaluated in 53 LUAD cell lines; none met the declared functional-dependency rule. This negative result must be retained and limits therapeutic-target claims.

## Enrichment analysis

- Primary GSEA found 1,058 positive terms at FDR < 0.05 (26 Hallmark and 1,032 GO-BP).
- Degree-residualized GSEA found 860 positive and 391 negative terms.
- The post hoc diagnostic hub/structural-gene-filtered analysis found 577 positive terms (20 Hallmark and 557 GO-BP).
- Immune genes and broad housekeeping genes are not removed. The diagnostic filter removes the declared top 5% STRING hubs, ribosomal/mitochondrial structural genes and the declared technical-housekeeping panel only where specified.
- Complete results are retained. Hallmark terms and nonredundant GO-BP representatives should be emphasized, while full GO tables belong in supplementary material.
- Local versioned GMT snapshots and SHA256 values are recorded in `05_interpretability/gene_set_manifest.json`.

## Figure format and submission position

All statistical figures have paired 300-dpi PNG and editable vector PDF files. The workflow figure is exported at 600 dpi as PNG/TIFF and as editable PDF/SVG. Figures use Arial and follow a 170-mm full-width publication layout where applicable.

The computational package is substantially stronger and internally consistent, but it does not guarantee acceptance by BMC Genomics. Its defensible contribution is an exploratory multiview, transductive gene-prioritization framework with explicit degree-bias diagnostics and multi-cohort computational characterization. The principal limitations are that simpler topology methods outperform the GNN, external survival did not replicate, DepMap dependency was negative, and no wet-lab validation is included.
