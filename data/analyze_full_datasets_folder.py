#!/usr/bin/env python3
"""
Analyze full datasets folder (Trnning dataset raw datset/datasets): all subfolders,
file types, and sample 1-5 files per folder to understand medical data types for training.

Output: report of data types so the training system can understand and use the data properly.
Does NOT read all files - only 1-5 samples per folder per extension.

Usage:
  python data/analyze_full_datasets_folder.py --root "path/to/datasets"
  python data/analyze_full_datasets_folder.py --out docs/DATASETS_FULL_ANALYSIS.json

  On Windows, if 'python' is not in PATH, use: py data/analyze_full_datasets_folder.py ...
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}
MAX_SAMPLES_PER_EXT = 5


def get_ext(name):
    n = name.lower()
    if n.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(n)[1].lower()


def infer_csv_type(header_line, sample_rows):
    """Infer: clinical, bbox/detection, sensor_timeseries, genomic, other."""
    h = (header_line or "").lower()
    if "xmin" in h and "ymin" in h and "class" in h:
        return "bbox_detection"
    if "age" in h and ("bmi" in h or "pain" in h or "depression" in h or "disease" in h):
        return "clinical_survey"
    if "gene" in h or "symbol" in h or "control group" in h or "test group" in h:
        return "genomic_expression"
    if any(x in h for x in ["hr", "bvp", "eda", "acc", "ibi", "temp"]) or (sample_rows and len(sample_rows) > 100 and "," not in (sample_rows[0] or "")):
        return "sensor_timeseries"
    return "tabular_other"


def sample_file(full_path, ext, rel_path):
    """Read 1-5 lines or header to infer type. Returns dict for report."""
    out = {"path": rel_path, "ext": ext}
    try:
        size = os.path.getsize(full_path)
        out["size_bytes"] = size
    except OSError:
        pass
    try:
        if ext == ".csv":
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[:6]
            if lines:
                out["header"] = lines[0].strip()[:200]
                out["row_sample"] = [l.strip()[:150] for l in lines[1:4] if l.strip()]
                out["inferred_type"] = infer_csv_type(lines[0], lines[1:])
        elif ext in (".txt",) and "data" in full_path.lower():
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[:5]
            if lines:
                out["first_line"] = lines[0].strip()[:200]
                out["inferred_type"] = "genomic_expression" if "gene" in lines[0].lower() or "group" in lines[0].lower() else "text"
        elif ext == ".json":
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read(800)
            if "categories" in raw and "annotations" in raw:
                out["inferred_type"] = "coco_annotations"
            else:
                out["inferred_type"] = "json_other"
        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"):
            out["inferred_type"] = "image_pil"
        elif ext == ".pkl":
            out["inferred_type"] = "pickle_sensor_or_model"
        elif ext == ".pdf":
            out["inferred_type"] = "document"
        elif ext == ".xlsx" or ext == ".xls":
            out["inferred_type"] = "spreadsheet"
        else:
            out["inferred_type"] = "other"
    except Exception as e:
        out["error"] = str(e)[:100]
    return out


def walk_and_analyze(root_dir):
    """Walk root_dir; for each dir with files, group by ext, sample 1-5 per ext."""
    root_dir = os.path.abspath(root_dir)
    report = {
        "root": root_dir,
        "folders": {},
        "modality_summary": defaultdict(list),
        "training_system_notes": []
    }
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == ".":
            rel_dir = ""
        files_only = [f for f in filenames if not f.startswith(".") and os.path.isfile(os.path.join(dirpath, f))]
        if not files_only:
            continue
        by_ext = defaultdict(list)
        for f in files_only:
            by_ext[get_ext(f)].append(f)
        key = rel_dir or "(root)"
        report["folders"][key] = {
            "file_extensions": {e: len(lst) for e, lst in by_ext.items()},
            "total_files": len(files_only),
            "samples": {}
        }
        for ext, lst in by_ext.items():
            sample_files = sorted(lst)[:MAX_SAMPLES_PER_EXT]
            report["folders"][key]["samples"][ext] = []
            for fn in sample_files:
                full = os.path.join(dirpath, fn)
                rel = os.path.join(rel_dir, fn) if rel_dir else fn
                report["folders"][key]["samples"][ext].append(sample_file(full, ext, rel))
        # Modality hint from path
        rl = rel_dir.lower()
        if "patient" in rl or "record" in rl:
            report["modality_summary"]["patient_records"].append(key)
        elif "medical imaging" in rl or "imaging" in rl or "endotect" in rl or "coco" in rl:
            report["modality_summary"]["medical_imaging"].append(key)
        elif "sensor" in rl or "wesad" in rl:
            report["modality_summary"]["sensor"].append(key)
        elif "genomic" in rl or "gene" in rl:
            report["modality_summary"]["genomic"].append(key)
        elif "pathology" in rl:
            report["modality_summary"]["pathology"].append(key)
    report["modality_summary"] = dict(report["modality_summary"])
    # Training system notes
    report["training_system_notes"].append("FedPINN expects: clinical.csv (age,bmi,pain,...) + us_embeddings.npy (128-d). Other modalities optional: genomic_data.npy (256), pathology_data.npy (64), sensor_data.npy (32).")
    report["training_system_notes"].append("clinical_survey CSVs can be normalized with data_loader.normalize_clinical_dataframe; map columns to CLINICAL_FEATURE_COLUMNS.")
    report["training_system_notes"].append("bbox_detection / coco_annotations: use for detection/segmentation training; images -> run image encoder -> us_embeddings.npy.")
    report["training_system_notes"].append("sensor_timeseries / pickle_sensor: convert to 32-d per sample for sensor_data.npy or use as auxiliary stream.")
    report["training_system_notes"].append("genomic_expression (e.g. Gene symbol, groups): convert to 256-d embeddings for genomic_data.npy.")
    return report


def main():
    ap = argparse.ArgumentParser(description="Analyze full datasets folder: file types and data types for training.")
    ap.add_argument("--root", default=None, help="Path to datasets folder (e.g. Trnning dataset raw datset/datasets)")
    ap.add_argument("--out", default=None, help="Write JSON report here")
    ap.add_argument("--quiet", action="store_true", help="Less console output")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    if args.root and os.path.isdir(args.root):
        root = os.path.abspath(args.root)
    else:
        candidates = [
            os.path.join(parent, "Trnning dataset raw datset", "datasets"),
            os.path.join(parent, "..", "Trnning dataset raw datset", "datasets"),
        ]
        root = None
        for c in candidates:
            if os.path.isdir(c):
                root = os.path.abspath(c)
                break
    if not root:
        print("Error: datasets folder not found. Use --root path/to/datasets")
        sys.exit(1)

    report = walk_and_analyze(root)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Report written to:", args.out)

    if not args.quiet:
        print("Root:", report["root"])
        print("\nModality hints (from path names):", report["modality_summary"])
        print("\nFolders (first 25):")
        for i, (k, v) in enumerate(report["folders"].items()):
            if i >= 25:
                print("  ... and", len(report["folders"]) - 25, "more folders")
                break
            print("  ", k, "->", v["file_extensions"], "| samples:", list(v["samples"].keys()))
        print("\nTraining system notes:")
        for n in report["training_system_notes"]:
            print("  -", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
