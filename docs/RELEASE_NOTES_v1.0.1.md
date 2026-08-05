# ctMV-GCN v1.0.1

This nomenclature-only release adopts the descriptive model name **ctMV-GCN** (cell-type-informed multiview graph convolutional network).

## Changed

- replaced the former internal model identifier with ctMV-GCN in source code, tables, captions and figures;
- labelled the primary curve explicitly as ctMV-GCN;
- corrected “weighted BCE” to “class-weighted CE”, matching the implemented two-logit `cross_entropy` loss;
- renamed model checkpoint and workflow-figure basenames consistently;
- introduced `CTMV_*` configuration variables while retaining `SCTDA_*` as backward-compatible aliases;
- updated manuscript-facing metadata and release documentation.

## Unchanged

- input data and graph construction;
- training, validation and test splits;
- model architecture and fitted numerical predictions;
- all reported performance, robustness and external-characterization values;
- interpretation boundaries and negative results.

The immutable v1.0.1 archive is available at https://doi.org/10.5281/zenodo.21807929. The previous v1.0.0 archive remains available at https://doi.org/10.5281/zenodo.21805985.
