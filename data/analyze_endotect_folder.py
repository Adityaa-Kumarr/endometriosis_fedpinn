#!/usr/bin/env python3
"""
Analyze EndoTect (or similar) test dataset folder: each subfolder's file types,
sample 2-5 files per folder, and report whether our system understands the data for training/predict.

Usage:
  python data/analyze_endotect_folder.py
  python data/analyze_endotect_folder.py --root "path/to/test dataset"
  python data/analyze_endotect_folder.py --root "path" --out report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

def get_ext(name):
    n = name.lower()
    if n.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(n)[1].lower()

def analyze_folder(root_dir, out_path=None):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return {"error": f"Not a directory: {root_dir}"}

    report = {
        "root": root_dir,
        "folders": {},
        "summary": {},
        "system_understands": {},
        "recommendations": []
    }

    try:
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except OSError as e:
        report["error"] = str(e)
        return report

    for sub in sorted(subdirs):
        subpath = os.path.join(root_dir, sub)
        by_ext = defaultdict(list)
        try:
            for f in os.listdir(subpath):
                fp = os.path.join(subpath, f)
                if os.path.isfile(fp) and not f.startswith("."):
                    ext = get_ext(f)
                    by_ext[ext].append(f)
        except OSError:
            by_ext = {}

        ext_counts = {e: len(files) for e, files in by_ext.items()}
        report["folders"][sub] = {
            "file_extensions": dict(ext_counts),
            "total_files": sum(ext_counts.values()),
            "samples": {}
        }

        # Sample 2-5 files per extension in this folder
        samples = report["folders"][sub]["samples"]
        for ext, files in by_ext.items():
            sample_files = sorted(files)[:5]
            samples[ext] = []
            for fn in sample_files:
                full = os.path.join(subpath, fn)
                entry = {"file": fn}
                try:
                    if ext == ".csv":
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()[:6]
                        entry["header"] = lines[0].strip() if lines else ""
                        entry["row_sample"] = [l.strip() for l in lines[1:4] if l.strip()]
                        entry["data_type"] = "tabular (bbox/labels)" if "xmin" in (lines[0] or "").lower() or "class" in (lines[0] or "").lower() else "tabular"
                    elif ext in (".txt", ".json"):
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read(500)
                        entry["preview"] = content[:300].replace("\n", " ")
                    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"):
                        entry["data_type"] = "image (PIL-readable)"
                        try:
                            from PIL import Image
                            with Image.open(full) as im:
                                entry["size"] = list(im.size)
                                entry["mode"] = im.mode
                        except Exception:
                            entry["size"] = os.path.getsize(full)
                    else:
                        entry["size_bytes"] = os.path.getsize(full)
                except Exception as e:
                    entry["error"] = str(e)
                samples[ext].append(entry)
        report["folders"][sub]["samples"] = samples

    # Interpret and system understanding
    for sub, info in report["folders"].items():
        exts = list(info["file_extensions"].keys())
        summary = []
        understands = []
        if ".csv" in exts:
            summary.append("CSV: object-detection bbox (class_name,xmin,ymin,xmax,ymax) or tabular")
            # Our training expects clinical CSV with age,bmi,... or bbox CSV for detection - different use
            understands.append("bbox CSV: not clinical columns; use for detection/segmentation training, not FedPINN clinical stream")
        if any(e in exts for e in [".jpg", ".jpeg", ".png"]):
            summary.append("Images: JPG/PNG (PIL-readable)")
            understands.append("images: app vision encoder can load these; training needs embedding pipeline (e.g. run encoder -> us_embeddings.npy)")
        if ".txt" in exts or ".json" in exts:
            summary.append("Text/JSON: metadata or labels")
        report["summary"][sub] = summary
        report["system_understands"][sub] = understands

    # Global recommendations
    has_images = any(".jpg" in report["folders"].get(s, {}).get("file_extensions", {}) or ".png" in report["folders"].get(s, {}).get("file_extensions", {}) for s in report["folders"])
    has_bbox = any(".csv" in report["folders"].get(s, {}).get("file_extensions", {}) for s in report["folders"])
    if has_images and has_bbox:
        report["recommendations"].append("EndoTect-style layout: images + bbox CSVs. To train: (1) Run image encoder on images -> 128-d embeddings; (2) Pair with clinical CSV per patient OR use bbox for detection loss; (3) FedPINN expects client folders with clinical.csv + us_embeddings.npy.")
    if has_images:
        report["recommendations"].append("Our app can use these images for prediction (upload -> vision encoder -> 128-d). For federated training, add a pipeline: images folder -> encoder -> us_embeddings.npy per sample.")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Analyze EndoTect/test dataset folder structure and data types.")
    ap.add_argument("--root", default=None, help="Path to test dataset folder")
    ap.add_argument("--out", default=None, help="Write JSON report here")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    if args.root and os.path.isdir(args.root):
        root = os.path.abspath(args.root)
    else:
        # Default: try common locations for EndoTect test dataset
        candidates = [
            os.path.join(parent, "Trnning dataset raw datset", "datasets", "medical imaging", "Dataset-3", "EndoTect", "test dataset"),
            os.path.join(parent, "..", "Trnning dataset raw datset", "datasets", "medical imaging", "Dataset-3", "EndoTect", "test dataset"),
        ]
        root = None
        for c in candidates:
            if os.path.isdir(c):
                root = os.path.abspath(c)
                break
    if not root:
        print("Error: folder not found. Use --root path/to/test dataset")
        sys.exit(1)

    report = analyze_folder(root, args.out)

    print("Root:", report.get("root", ""))
    print()
    for sub, info in report.get("folders", {}).items():
        print("=" * 60)
        print("FOLDER:", sub)
        print("=" * 60)
        print("  File types:", info.get("file_extensions", {}))
        print("  Total files:", info.get("total_files", 0))
        print("  Summary:", report.get("summary", {}).get(sub, []))
        print("  System understands:", report.get("system_understands", {}).get(sub, []))
        print("  Sample files (2-5 per type):")
        for ext, samples in info.get("samples", {}).items():
            for s in samples[:3]:
                print(f"    {ext}: {s.get('file', '')} -> {s.get('data_type', s.get('header', s.get('size', '')))}")
        print()
    print("Recommendations:", report.get("recommendations", []))
    if args.out:
        print("Report written to:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
