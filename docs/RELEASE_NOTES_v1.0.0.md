# scTDA-GCN v1.0.0

This is the reproducible manuscript release.

## Included

- all custom Python source files used for the reported analyses;
- pinned Python dependencies and a portable local configuration example;
- exact input filenames, access routes, byte counts and SHA-256 checksums;
- an input-validation command (`python src/run_all.py --check-inputs`);
- frozen publication-level tables and figures;
- full benchmark, degree/topology robustness, biological characterization and negative validation results;
- MIT code license, citation metadata and Zenodo metadata.

## Excluded by design

- raw or access-controlled third-party datasets;
- patient-level TCGA working tables and external API caches;
- trained weights and intermediate graph tensors.

## Interpretation boundary

The release supports computational prioritization and hypothesis generation. The Open Targets endpoint denotes existing clinical-stage evidence, RWR and degree baselines outperform the GNN in the primary benchmark, and no robust candidate satisfies the prespecified DepMap LUAD dependency rule.
