# Raw input layout

Raw third-party datasets are not distributed in this repository. Download them from the sources listed in [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

The recommended layout is:

```text
data/
├── README.md
├── core/
│   ├── scRNA_multichannel_features_TISCH.csv
│   ├── 9606.protein.info.v12.0.txt/
│   │   └── 9606.protein.info.v12.0.txt
│   ├── 9606.protein.links.v12.0.txt/
│   │   └── 9606.protein.links.v12.0.txt
│   ├── c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt
│   ├── OT-EFO_0000571-associated-targets-2026_3_23-v0_0.tsv
│   ├── TCGA.LUAD.sampleMap_HiSeqV2/
│   │   └── HiSeqV2
│   ├── TCGA.LUAD.sampleMap_LUAD_clinicalMatrix
│   ├── DepMap_Public_26Q1_CRISPRGeneEffect.csv
│   └── DepMap_Public_26Q1_Model.csv
└── external_validation/
    ├── GSE68465_series_matrix.txt.gz
    ├── GPL96-57554.txt
    └── DepMap_Public_26Q1_LUAD_top20_CRISPR.csv
```

Some source portals can change export filenames. If a downloaded file has a different name, either rename it to the name expected by `src/config.py` or update a local configuration without committing machine-specific paths.

