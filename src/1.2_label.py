from __future__ import annotations

import shutil

import pandas as pd

from common import save_json
from config import (
    CLINICAL_PHASE_THRESHOLD,
    LABEL_FILE,
    OPEN_TARGETS_FILE,
    PACKAGED_LABEL_AUDIT_FILE,
    PACKAGED_LABEL_FILE,
    ensure_result_dirs,
)


def main() -> None:
    ensure_result_dirs()
    if not OPEN_TARGETS_FILE.exists():
        if not PACKAGED_LABEL_FILE.exists() or not PACKAGED_LABEL_AUDIT_FILE.exists():
            raise FileNotFoundError(
                "Open Targets export is missing and the frozen packaged label snapshot is unavailable: "
                f"{OPEN_TARGETS_FILE}"
            )
        labels = pd.read_csv(PACKAGED_LABEL_FILE)
        labels.to_csv(LABEL_FILE, index=False, encoding="utf-8-sig")
        shutil.copyfile(
            PACKAGED_LABEL_AUDIT_FILE,
            LABEL_FILE.with_name("open_targets_label_audit.csv"),
        )
        save_json({
            "source": "frozen repository snapshot results/01_data/clinical_target_labels.csv",
            "original_source": "Open Targets LUAD EFO_0000571 export dated 2026-03-23",
            "disease_id": "EFO_0000571",
            "positive_definition": "maxClinicalTrialPhase > 0",
            "positive_gene_count": int(labels["symbol"].nunique()),
            "global_score_is_not_used_as_the_therapeutic_label": True,
        }, LABEL_FILE.with_name("label_definition.json"))
        print(f"Clinical-evidence positive genes: {labels['symbol'].nunique():,} (frozen snapshot)")
        return

    frame = pd.read_csv(OPEN_TARGETS_FILE, sep="\t")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["maxClinicalTrialPhase"] = pd.to_numeric(frame["maxClinicalTrialPhase"], errors="coerce")
    frame["globalScore"] = pd.to_numeric(frame["globalScore"], errors="coerce")
    frame["chembl"] = pd.to_numeric(frame["chembl"], errors="coerce")
    frame = frame.dropna(subset=["symbol"]).drop_duplicates("symbol", keep="first")
    positive = frame["maxClinicalTrialPhase"].fillna(0) > CLINICAL_PHASE_THRESHOLD
    labels = frame.loc[positive, ["symbol", "maxClinicalTrialPhase", "chembl", "globalScore"]].copy()
    labels["label_definition"] = "Open Targets LUAD association with maxClinicalTrialPhase > 0"
    labels = labels.sort_values(["maxClinicalTrialPhase", "globalScore"], ascending=False)
    labels.to_csv(LABEL_FILE, index=False, encoding="utf-8-sig")
    frame[["symbol", "globalScore", "maxClinicalTrialPhase", "chembl"]].to_csv(
        LABEL_FILE.with_name("open_targets_label_audit.csv"), index=False, encoding="utf-8-sig"
    )
    save_json({
        "source": str(OPEN_TARGETS_FILE),
        "disease_id": "EFO_0000571",
        "positive_definition": "maxClinicalTrialPhase > 0",
        "positive_gene_count": int(labels["symbol"].nunique()),
        "global_score_is_not_used_as_the_therapeutic_label": True,
    }, LABEL_FILE.with_name("label_definition.json"))
    print(f"Clinical-evidence positive genes: {labels['symbol'].nunique():,}")


if __name__ == "__main__":
    main()
