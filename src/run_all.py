from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import LOG_DIR, PROJECT_ROOT, RESULT_ROOT, ensure_result_dirs


COMMANDS = [
    ("1.1_ppi.py",),
    ("1.2_label.py",),
    ("1.3_coexp.py",),
    ("1.4_pathway.py",),
    ("1.5_data.py",),
    ("2.1_compare3.py",),
    ("2.2_train2.py",),
    ("2.3_predict2.py",),
    ("3.1_top-k3.py",),
    ("3.3_biological_validation.py", "--degree-only"),
    ("4.1_robustness.py",),
    ("3.3_biological_validation.py", "--candidate-validation-only"),
    ("3.2_analysis3.py",),
    ("3.4_TCGA3.py",),
    ("4.2_external_validation.py",),
    ("4.3_depmap_validation.py",),
    ("5.1_workflow_figure.py",),
    ("5.2_framework_schematic.py",),
]


def clear_final_results() -> None:
    resolved = RESULT_ROOT.resolve()
    if resolved.name != "results_final":
        raise ValueError(
            f"Refusing --fresh because result_root is not named results_final: {resolved}"
        )
    if resolved == PROJECT_ROOT.resolve() or resolved.parent == resolved:
        raise ValueError(f"Unsafe result root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete only the configured directory named results_final before running.",
    )
    parser.add_argument(
        "--check-inputs", action="store_true",
        help="Validate required input files and frozen SHA-256 checksums, then exit.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.check_inputs:
        raise SystemExit(subprocess.run([sys.executable, str(root / "check_inputs.py")], cwd=root).returncode)
    if args.fresh:
        clear_final_results()
    ensure_result_dirs()
    for command in COMMANDS:
        script, *arguments = command
        label = " ".join(command)
        print(f"\n===== Running {label} =====", flush=True)
        completed = subprocess.run(
            [sys.executable, str(root / script), *arguments],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_suffix = "_".join(arg.lstrip("-").replace("-", "_") for arg in arguments)
        log_name = f"{script}{'_' + log_suffix if log_suffix else ''}.log"
        (LOG_DIR / log_name).write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, flush=True)
        if completed.returncode != 0:
            raise SystemExit(f"{label} failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
