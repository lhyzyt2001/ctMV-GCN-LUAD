import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
PACKAGED_RESULTS_ROOT = REPOSITORY_ROOT / "results"
LOCAL_CONFIG_FILE = Path(os.environ.get("SCTDA_CONFIG", PROJECT_ROOT / "config.local.json"))
if LOCAL_CONFIG_FILE.exists():
    LOCAL_CONFIG = json.loads(LOCAL_CONFIG_FILE.read_text(encoding="utf-8"))
else:
    LOCAL_CONFIG = {}


def configured_path(key: str, environment_variable: str, default: Path) -> Path:
    value = os.environ.get(environment_variable, LOCAL_CONFIG.get(key))
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_ROOT = configured_path("data_root", "SCTDA_DATA_ROOT", REPOSITORY_ROOT / "data" / "core")
SOURCE_ROOT = configured_path("source_root", "SCTDA_SOURCE_ROOT", DATA_ROOT)
RESULT_ROOT = configured_path("result_root", "SCTDA_RESULT_ROOT", REPOSITORY_ROOT / "results_final")
EXTERNAL_DATA_ROOT = configured_path(
    "external_data_root", "SCTDA_EXTERNAL_DATA_ROOT",
    REPOSITORY_ROOT / "data" / "external_validation",
)

DATA_DIR = RESULT_ROOT / "01_data"
BENCHMARK_DIR = RESULT_ROOT / "02_benchmark"
MODEL_DIR = RESULT_ROOT / "03_model"
TOPK_DIR = RESULT_ROOT / "04_topk"
INTERPRET_DIR = RESULT_ROOT / "05_interpretability"
TCGA_DIR = RESULT_ROOT / "06_tcga"
VALIDATION_DIR = RESULT_ROOT / "07_validation"
ROBUSTNESS_DIR = RESULT_ROOT / "08_robustness"
LOG_DIR = RESULT_ROOT / "logs"

CONFIGURED_FEATURE_FILE = DATA_ROOT / "scRNA_multichannel_features_TISCH.csv"
PACKAGED_FEATURE_FILE = PACKAGED_RESULTS_ROOT / "01_data" / "celltype_expression_features.csv"
FEATURE_FILE = (
    CONFIGURED_FEATURE_FILE
    if CONFIGURED_FEATURE_FILE.exists() or not PACKAGED_FEATURE_FILE.exists()
    else PACKAGED_FEATURE_FILE
)
STRING_INFO_FILE = DATA_ROOT / "9606.protein.info.v12.0.txt" / "9606.protein.info.v12.0.txt"
STRING_LINK_FILE = DATA_ROOT / "9606.protein.links.v12.0.txt" / "9606.protein.links.v12.0.txt"
KEGG_GMT_FILE = DATA_ROOT / "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt"
OPEN_TARGETS_FILE = SOURCE_ROOT / "OT-EFO_0000571-associated-targets-2026_3_23-v0_0.tsv"
PACKAGED_LABEL_FILE = PACKAGED_RESULTS_ROOT / "01_data" / "clinical_target_labels.csv"
PACKAGED_LABEL_AUDIT_FILE = PACKAGED_RESULTS_ROOT / "01_data" / "open_targets_label_audit.csv"
TCGA_EXPRESSION_FILE = DATA_ROOT / "TCGA.LUAD.sampleMap_HiSeqV2" / "HiSeqV2"
TCGA_CLINICAL_FILE = DATA_ROOT / "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix"
DEPMAP_GENE_EFFECT_FILE = configured_path(
    "depmap_gene_effect_file", "SCTDA_DEPMAP_GENE_EFFECT_FILE",
    DATA_ROOT / "DepMap_Public_26Q1_CRISPRGeneEffect.csv",
)
DEPMAP_MODEL_FILE = configured_path(
    "depmap_model_file", "SCTDA_DEPMAP_MODEL_FILE",
    DATA_ROOT / "DepMap_Public_26Q1_Model.csv",
)
DEPMAP_CUSTOM_FILE = configured_path(
    "depmap_custom_file", "SCTDA_DEPMAP_CUSTOM_FILE",
    EXTERNAL_DATA_ROOT / "DepMap_Public_26Q1_LUAD_top20_CRISPR.csv",
)

NODE_MAPPING_FILE = DATA_DIR / "node_mapping.json"
PPI_EDGE_FILE = DATA_DIR / "edge_index_string.npy"
PPI_WEIGHT_FILE = DATA_DIR / "edge_weight_string.npy"
COEXP_EDGE_FILE = DATA_DIR / "edge_index_celltype_similarity.npy"
COEXP_WEIGHT_FILE = DATA_DIR / "edge_weight_celltype_similarity.npy"
PATHWAY_EDGE_FILE = DATA_DIR / "edge_index_pathway.npy"
PATHWAY_WEIGHT_FILE = DATA_DIR / "edge_weight_pathway.npy"
LABEL_FILE = DATA_DIR / "clinical_target_labels.csv"
RAW_DATASET_FILE = DATA_DIR / "GNN_Dataset_Raw.pt"

SEED = 42
PCA_COMPONENTS = 5
STRING_SCORE_THRESHOLD = 700
CELLTYPE_K = 12
PATHWAY_K = 12

# maxClinicalTrialPhase is normalized by Open Targets to (0, 1]; any value > 0
# denotes at least one clinical-stage drug programme for the LUAD association.
CLINICAL_PHASE_THRESHOLD = 0.0

N_SPLITS = 5
N_REPEATS = 3
INNER_VAL_FRACTION = 0.15
MAX_EPOCHS = 50
PATIENCE = 4
VALIDATION_INTERVAL = 5
BOOTSTRAP_REPLICATES = 2000
FINAL_ENSEMBLE_SEEDS = tuple(range(42, 142, 10))


def ensure_result_dirs() -> None:
    for directory in (
        RESULT_ROOT,
        DATA_DIR,
        BENCHMARK_DIR,
        MODEL_DIR,
        TOPK_DIR,
        INTERPRET_DIR,
        TCGA_DIR,
        VALIDATION_DIR,
        ROBUSTNESS_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
