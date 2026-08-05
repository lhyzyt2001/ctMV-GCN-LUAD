# Raw input layout

Raw third-party datasets are not distributed in this repository. Download them from the authoritative sources and follow the version notes in [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md). Exact study-file checksums are in [`../docs/INPUT_CHECKSUMS.tsv`](../docs/INPUT_CHECKSUMS.tsv).

The expected layout is:

```text
data/
├── README.md
├── core/
│   ├── scRNA_multichannel_features_TISCH.csv        # optional; packaged fallback is included
│   ├── 9606.protein.info.v12.0.txt/
│   │   └── 9606.protein.info.v12.0.txt
│   ├── 9606.protein.links.v12.0.txt/
│   │   └── 9606.protein.links.v12.0.txt
│   ├── c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt
│   ├── OT-EFO_0000571-associated-targets-2026_3_23-v0_0.tsv  # optional; frozen-label fallback is included
│   ├── TCGA.LUAD.sampleMap_HiSeqV2/
│   │   └── HiSeqV2
│   ├── TCGA.LUAD.sampleMap_LUAD_clinicalMatrix
│   ├── DepMap_Public_26Q1_CRISPRGeneEffect.csv       # optional full-release route
│   └── DepMap_Public_26Q1_Model.csv                  # optional full-release route
└── external_validation/
    ├── GSE68465_series_matrix.txt.gz
    ├── GPL96.annot.gz
    └── DepMap_Public_26Q1_LUAD_top20_CRISPR.csv     # used route; replaces the two full DepMap files
```

The packaged TISCH feature matrix and Open Targets label snapshots under `results/01_data` are sufficient replacements for the two inputs marked as optional. The STRING, MSigDB/KEGG, TCGA, GEO and DepMap inputs must still be downloaded under their source terms.

From the repository root, copy `src/config.local.example.json` to `src/config.local.json`, adjust paths if needed, and validate every input before starting the long pipeline:

```bash
python src/run_all.py --check-inputs
```

Relative paths in `config.local.json` are resolved against the `src` directory, so the distributed example works independently of the shell's current directory. If a source portal changes an export filename, rename it to the documented name or update only your untracked local configuration.
