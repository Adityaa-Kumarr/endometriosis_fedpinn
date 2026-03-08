#!/usr/bin/env python3
"""
Analyze raw / training dataset folder: ALL file extensions and IMAGING dataset types.

Scans a dataset root (e.g. dataset/, raw/) and reports:
- Every file extension found (with counts)
- Which of these are IMAGE/imaging types (so the system can support them for train & predict)
- A summary and optional JSON report for the app to support these image types

Usage:
  python data/analyze_raw_dataset_extensions.py
  python data/analyze_raw_dataset_extensions.py --root path/to/raw
  python data/analyze_raw_dataset_extensions.py --root dataset --out report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Double extensions (checked first, e.g. .nii.gz)
DOUBLE_EXTENSIONS = ('.nii.gz',)

# Known IMAGING/IMAGE file types (medical + general) — so we can support them in app & encoder
IMAGING_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif',
    '.dcm', '.dicom',           # DICOM
    '.nii', '.nii.gz', '.nrrd', '.nhrd', '.nhdr',  # NIfTI, NRRD
    '.mha', '.mhd', '.raw',   # MetaImage (often used with .raw)
    '.png', '.jp2', '.j2k',   # JPEG 2000
    '.svs', '.tif',           # whole slide (pathology)
})

# Extensions the current app + PIL can open as images (for vision encoder)
PIL_READABLE_IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif',
})


def get_extension(filename):
    """Return lower-case extension, handling double extensions like .nii.gz."""
    name = filename.lower()
    for double in DOUBLE_EXTENSIONS:
        if name.endswith(double):
            return double
    ext = os.path.splitext(name)[1].lower()
    return ext or '(none)'


def scan_all_extensions(root_dir, exclude_dirs=None):
    """Walk root_dir and return (extension -> list of relative paths), (extension -> count)."""
    exclude_dirs = set(exclude_dirs or ['.git', '__pycache__', 'node_modules', '.venv', 'venv'])
    root_dir = os.path.abspath(root_dir)
    by_ext = defaultdict(list)
    count_by_ext = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == '.':
            rel_dir = ''
        for f in filenames:
            if f.startswith('.'):
                continue
            ext = get_extension(f)
            rel_path = os.path.join(rel_dir, f) if rel_dir else f
            by_ext[ext].append(rel_path)
            count_by_ext[ext] += 1

    return dict(by_ext), dict(count_by_ext)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze raw/training dataset folder: all file extensions and imaging types.'
    )
    parser.add_argument('--root', default=None, help='Dataset root (default: dataset/ then raw/ next to script)')
    parser.add_argument('--out', default=None, help='Write extension + imaging report to this JSON file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    if args.root is None:
        root_dir = os.path.join(parent, 'dataset')
        for candidate in ('dataset', 'raw', 'data'):
            c = os.path.join(parent, candidate)
            if os.path.isdir(c):
                root_dir = c
                break
    else:
        root_dir = args.root

    if not os.path.isdir(root_dir):
        print(f"Error: Not a directory: {root_dir}")
        sys.exit(1)

    by_ext, count_by_ext = scan_all_extensions(root_dir)
    total_files = sum(count_by_ext.values())

    # Imaging: which extensions we found that are in IMAGING_EXTENSIONS
    imaging_found = {ext: count_by_ext[ext] for ext in by_ext if ext in IMAGING_EXTENSIONS}
    # PIL-readable (can be fed to vision encoder today)
    pil_ready = {ext: count_by_ext[ext] for ext in by_ext if ext in PIL_READABLE_IMAGE_EXTENSIONS}
    # Medical-only (dcm, nii, etc.) — need special loaders for full support
    medical_imaging = {ext: count_by_ext[ext] for ext in by_ext
                      if ext in IMAGING_EXTENSIONS and ext not in PIL_READABLE_IMAGE_EXTENSIONS}

    # ---- Console report ----
    print(f"Root: {os.path.abspath(root_dir)}")
    print(f"Total files: {total_files}")
    print()
    print("=" * 60)
    print("ALL FILE EXTENSIONS FOUND (by count)")
    print("=" * 60)
    for ext in sorted(count_by_ext.keys(), key=lambda e: (-count_by_ext[e], e)):
        print(f"  {ext or '(no ext)'}: {count_by_ext[ext]} file(s)")
    print()
    print("=" * 60)
    print("IMAGING / IMAGE DATASET TYPES IN THIS FOLDER")
    print("=" * 60)
    if imaging_found:
        for ext in sorted(imaging_found.keys()):
            count = imaging_found[ext]
            pil = " (PIL-readable -> vision encoder)" if ext in PIL_READABLE_IMAGE_EXTENSIONS else " (medical -> needs loader)"
            print(f"  {ext}: {count} file(s){pil}")
        print()
        print("PIL-readable (supported in app today for upload + vision):", sorted(pil_ready.keys()))
        if medical_imaging:
            print("Medical imaging (add loader for full support):", sorted(medical_imaging.keys()))
    else:
        print("  (no imaging extensions found in this folder)")
    print()
    print("=" * 60)
    print("RECOMMENDATION FOR TRAIN & PREDICT")
    print("=" * 60)
    all_imaging_exts = sorted(set(imaging_found.keys()))
    if all_imaging_exts:
        print("  Image extensions to support in the system:", all_imaging_exts)
        print("  Already supported in app (upload + vision):", sorted(PIL_READABLE_IMAGE_EXTENSIONS & set(by_ext)))
        if medical_imaging:
            print("  Add support for (e.g. pydicom, nibabel):", sorted(medical_imaging.keys()))
    else:
        print("  No image files in this dataset; no change needed for imaging.")
    print()

    # JSON report for downstream (app / CI)
    report = {
        'root': os.path.abspath(root_dir),
        'total_files': total_files,
        'all_extensions': {ext: count_by_ext[ext] for ext in sorted(count_by_ext.keys())},
        'imaging_extensions_found': imaging_found,
        'imaging_extensions_to_support': all_imaging_exts,
        'pil_readable_imaging': list(pil_ready.keys()),
        'medical_imaging_extensions': list(medical_imaging.keys()),
        'known_imaging_list': list(IMAGING_EXTENSIONS),
    }
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Report written to: {args.out}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
