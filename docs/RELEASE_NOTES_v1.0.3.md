# ctMV-GCN v1.0.3

This final manuscript-sensitivity release archives the additional diagnostics used by the submitted article.

## Added

- portable code for per-covariate Schoenfeld-residual proportional-hazards diagnostics;
- full TCGA-LUAD and GSE68465 PH diagnostic outputs;
- a 2,000-replicate STRING-degree-matched gene-cluster bootstrap retaining each sampled gene across all three OOF repeats;
- a code-derived reproducibility-parameter audit;
- updated Additional files 1 and 4 and new Additional files 5 and 6.

## Interpretation update

- the gene-cluster bootstrap produced a ctMV-GCN-minus-degree AUPRC difference of 0.0215 (95% CI -0.0018 to 0.0444; P = 0.072), so the evidence for an advantage beyond degree is directional but not statistically robust;
- expression-term PH violations were identified for APOA2 and FCER1A in TCGA overall survival and for FCER1A, POSTN and C4A in GSE68465; affected HRs are interpreted as average associations over follow-up.

## Unchanged

- source graph, labels and cross-validation splits;
- model architecture, fitted predictions and candidate ranking;
- primary benchmark, expression, protein and DepMap numerical results.

Earlier releases and DOIs remain immutable, including v1.0.2 at https://doi.org/10.5281/zenodo.21870287.
