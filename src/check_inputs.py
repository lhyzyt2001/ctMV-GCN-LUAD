from __future__ import annotations

import hashlib
from pathlib import Path

from config import (
    CONFIGURED_FEATURE_FILE,
    DEPMAP_CUSTOM_FILE,
    DEPMAP_GENE_EFFECT_FILE,
    DEPMAP_MODEL_FILE,
    EXTERNAL_DATA_ROOT,
    KEGG_GMT_FILE,
    OPEN_TARGETS_FILE,
    PACKAGED_FEATURE_FILE,
    PACKAGED_LABEL_AUDIT_FILE,
    PACKAGED_LABEL_FILE,
    STRING_INFO_FILE,
    STRING_LINK_FILE,
    TCGA_CLINICAL_FILE,
    TCGA_EXPRESSION_FILE,
)


EXPECTED_SHA256 = {
    "scRNA_multichannel_features_TISCH.csv": "d9e6398e0412f3fc25be7976097602fbcb148481041d2f09a97f7f8cd6283426",
    "celltype_expression_features.csv": "2772f10ef57b6bfcbb9d650d76f853b7b3932c0fea1c71baac467f4fd2e56c8a",
    "9606.protein.info.v12.0.txt": "a65e52748022d6a8e953843645a5032b094017225d95de9c6cb6fd4577d0a7ad",
    "9606.protein.links.v12.0.txt": "b83b0304f72455c82a05294713e4e2c6ad38338281b07f433da14d69943c6483",
    "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt": "e257052d604cb2f386f8372b3fb9b494095688ac8ae0d72ba947cc00f5628cb1",
    "OT-EFO_0000571-associated-targets-2026_3_23-v0_0.tsv": "f5c62d804acc7887d2a473358130373035bceb426df873e1bb3d07117748250b",
    "clinical_target_labels.csv": "08b94664f5ceb82f42946c637e2a2918918d37dbca47c5f8dc3b0efac9af1edb",
    "open_targets_label_audit.csv": "cb61b3ffbce0de0a207ebe0ee0aef0fd0289f3743f542b18d901a8e9a918f0c4",
    "HiSeqV2": "64421a30147a689571f4cc537eb484b294e4325807e2b83031e7d918e07fe83c",
    "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix": "0e7a7fa7773a44326d2270160201b059c50c8a0c75359359a8639772497db59a",
    "GSE68465_series_matrix.txt.gz": "f4a74ef436302e016029df8cea8e01a1a36c98d087501c78668e05727d38bdbf",
    "GPL96.annot.gz": "88e0b22362bac779eb220b3b185c80faa6510a92b9358eaad159a561ab4351c4",
    "DepMap_Public_26Q1_LUAD_top20_CRISPR.csv": "9337a5e5b2bbb5bd275a19de1216e208a7f8ed9ef566bf43b17875aa809bbb7c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_one(label: str, path: Path, expected: str | None = None) -> bool:
    if not path.exists():
        print(f"MISSING  {label}: {path}")
        return False
    actual = sha256(path) if expected else None
    if expected and actual != expected:
        print(f"MISMATCH {label}: {path}")
        print(f"         expected {expected}")
        print(f"         observed {actual}")
        return False
    suffix = f" sha256={actual}" if actual else ""
    print(f"OK       {label}: {path}{suffix}")
    return True


def main() -> int:
    ok = True

    feature = CONFIGURED_FEATURE_FILE if CONFIGURED_FEATURE_FILE.exists() else PACKAGED_FEATURE_FILE
    feature_hash = EXPECTED_SHA256.get(feature.name)
    ok &= check_one("cell-type expression features", feature, feature_hash)

    if OPEN_TARGETS_FILE.exists():
        ok &= check_one("Open Targets LUAD export", OPEN_TARGETS_FILE, EXPECTED_SHA256[OPEN_TARGETS_FILE.name])
    else:
        ok &= check_one("frozen Open Targets positive labels", PACKAGED_LABEL_FILE, EXPECTED_SHA256[PACKAGED_LABEL_FILE.name])
        ok &= check_one("frozen Open Targets label audit", PACKAGED_LABEL_AUDIT_FILE, EXPECTED_SHA256[PACKAGED_LABEL_AUDIT_FILE.name])

    required = [
        ("STRING protein information", STRING_INFO_FILE),
        ("STRING functional associations", STRING_LINK_FILE),
        ("MSigDB KEGG Medicus gene sets", KEGG_GMT_FILE),
        ("TCGA-LUAD expression", TCGA_EXPRESSION_FILE),
        ("TCGA-LUAD clinical matrix", TCGA_CLINICAL_FILE),
        ("GSE68465 series matrix", EXTERNAL_DATA_ROOT / "GSE68465_series_matrix.txt.gz"),
        ("GPL96 annotation", EXTERNAL_DATA_ROOT / "GPL96.annot.gz"),
    ]
    for label, path in required:
        ok &= check_one(label, path, EXPECTED_SHA256.get(path.name))

    if DEPMAP_CUSTOM_FILE.exists():
        ok &= check_one(
            "DepMap Public 26Q1 LUAD custom download",
            DEPMAP_CUSTOM_FILE,
            EXPECTED_SHA256.get(DEPMAP_CUSTOM_FILE.name),
        )
    elif DEPMAP_GENE_EFFECT_FILE.exists() and DEPMAP_MODEL_FILE.exists():
        ok &= check_one("DepMap Public 26Q1 CRISPRGeneEffect", DEPMAP_GENE_EFFECT_FILE)
        ok &= check_one("DepMap Public 26Q1 Model metadata", DEPMAP_MODEL_FILE)
    else:
        print(
            "MISSING  DepMap input: provide the custom LUAD download or both "
            f"{DEPMAP_GENE_EFFECT_FILE} and {DEPMAP_MODEL_FILE}"
        )
        ok = False

    print("\nInput validation " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
