# ctMV-GCN v1.0.2

This manuscript-alignment release adds the exact submission figures and supplementary workbooks used by the final article draft.

## Added

- the portable final plotting script `src/6.1_bmc_submission_figures.py`;
- independent PDF and PNG files for Figures 1-10 under `results/manuscript_submission`;
- four formally numbered, machine-readable supplementary workbooks under `results/manuscript_submission/Additional_files`;
- a submission manifest with file sizes and SHA-256 checksums.

## Clarified

- Figure 1 now uses **Clinical-evidence score**, because the softmax output was not probability-calibrated;
- the repeat-aware bootstrap sampling scheme is described explicitly;
- residual clinical confounding and the interpretation boundary of negative DepMap results are stated in the manuscript;
- Additional files 1-4 are cited sequentially and described by filename, format, title and content.

## Unchanged

- source datasets and graph construction;
- cross-validation splits, model architecture and fitted predictions;
- all reported benchmark, robustness and external-characterization values;
- the negative survival-replication and DepMap findings.

The immutable earlier releases remain available at:

- v1.0.1: https://doi.org/10.5281/zenodo.21807929
- v1.0.0: https://doi.org/10.5281/zenodo.21805985

Zenodo will assign the v1.0.2 version-specific DOI after this GitHub release is published.
