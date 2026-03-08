#!/usr/bin/env python3
"""
Analyze Training Dataset — folder structure, file formats, and data types.

Scans a dataset root (default: dataset/ or raw/) and reports:
- Folder structure and file counts
- Per-file: format, inferred data type (clinical, ultrasound, genomic, pathology, sensor, etc.)
- Required columns present/missing for clinical CSVs
- A machine-readable manifest (JSON) so the training system knows what data can be used

Usage:
  python data/analyze_training_dataset.py
  python data/analyze_training_dataset.py --root path/to/raw/datasets
  python data/analyze_training_dataset.py --root dataset --out manifest.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Optional deps (graceful if missing)
try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None

# Project canonical schema (align with data_loader)
CLINICAL_FEATURE_COLUMNS = [
    'age', 'bmi', 'pelvic_pain_score', 'dysmenorrhea_score', 'dyspareunia',
    'family_history', 'ca125', 'estradiol', 'progesterone'
]
LABEL_COLUMN = 'label'
STAGE_COLUMN = 'stage'
ALTERNATIVE_LABEL_COLUMN = 'endometriosis_present'

# Expected embedding dimensions per modality (for .npy inference)
MODALITY_DIMS = {
    128: 'ultrasound',
    256: 'genomic',
    64: 'pathology',
    32: 'sensor',
}

# Filename hints for modality (when shape alone is ambiguous)
MODALITY_FILE_HINTS = {
    'us_embeddings': 'ultrasound',
    'ultrasound': 'ultrasound',
    'genomic_data': 'genomic',
    'genomic': 'genomic',
    'pathology_data': 'pathology',
    'pathology': 'pathology',
    'sensor_data': 'sensor',
    'sensor': 'sensor',
}


def normalize_columns(df_columns):
    """Lowercase and strip column names."""
    return [str(c).lower().strip() for c in df_columns]


def infer_csv_data_type(path, columns):
    """
    Infer data type of a CSV from path and column names.
    Returns: 'clinical' | 'sensor' | 'genomic' | 'pathology' | 'imaging_meta' | 'unknown'
    """
    cols_lower = [str(c).lower() for c in columns]
    path_lower = path.lower()

    # Clinical: must have several of these
    clinical_keywords = ['age', 'bmi', 'pelvic_pain', 'dysmenorrhea', 'ca125', 'estradiol', 'progesterone', 'stage', 'label']
    clinical_score = sum(1 for k in clinical_keywords if any(k in c for c in cols_lower))

    # Sensor / wearable
    sensor_keywords = ['accel', 'gyro', 'heart_rate', 'hr', 'eda', 'ecg', 'respiration', 'step', 'temp', 'sensor', 'timestamp']
    sensor_score = sum(1 for k in sensor_keywords if any(k in c for c in cols_lower))

    # Genomic
    genomic_keywords = ['gene', 'expression', 'rna', 'dna', 'snp', 'variant', 'ct_value', 'dct']
    genomic_score = sum(1 for k in genomic_keywords if any(k in c for c in cols_lower)) or (1 if 'genom' in path_lower or 'gene' in path_lower else 0)

    # Pathology / microbiome
    path_keywords = ['microbiome', 'bacteria', 'taxon', 'abundance', 'biopsy', 'histology', 'slide']
    path_score = sum(1 for k in path_keywords if any(k in c for c in cols_lower)) or (1 if 'path' in path_lower or 'microb' in path_lower else 0)

    # Imaging list (file paths, image IDs)
    imaging_keywords = ['image', 'dicom', 'mri', 'us', 'scan', 'patient_id', 'series']
    imaging_score = sum(1 for k in imaging_keywords if any(k in c for c in cols_lower))

    if clinical_score >= 3 and sensor_score <= 1:
        return 'clinical'
    if sensor_score >= 2 and clinical_score < 3:
        return 'sensor'
    if genomic_score >= 1:
        return 'genomic'
    if path_score >= 1:
        return 'pathology'
    if imaging_score >= 2:
        return 'imaging_meta'
    return 'unknown'


def check_clinical_readiness(columns):
    """Check which canonical columns are present; return (present, missing)."""
    norm = set(normalize_columns(columns))
    # Accept alternates
    if 'pelvic_pain' in norm and 'pelvic_pain_score' not in norm:
        norm.add('pelvic_pain_score')
    if 'dysmenorrhea' in norm and 'dysmenorrhea_score' not in norm:
        norm.add('dysmenorrhea_score')
    if 'ca-125' in norm and 'ca125' not in norm:
        norm.add('ca125')
    required = set(CLINICAL_FEATURE_COLUMNS)
    present = required & norm
    missing = required - norm
    return sorted(present), sorted(missing)


def infer_npy_modality(path, shape):
    """Infer modality from file path and array shape."""
    path_lower = path.lower()
    for hint, mod in MODALITY_FILE_HINTS.items():
        if hint in path_lower:
            return mod
    if len(shape) == 2:
        dim = shape[1]
        return MODALITY_DIMS.get(dim, f'embedding_{dim}d')
    return 'array'


def _get_ext(name):
    """Get lower-case extension, handling double extensions like .nii.gz."""
    n = name.lower()
    if n.endswith('.nii.gz'):
        return '.nii.gz'
    return os.path.splitext(n)[1].lower()


def analyze_file(root_dir, rel_path, full_path, results):
    """Analyze a single file; append to results list and return inferred type."""
    rel = rel_path.replace('\\', '/')
    name = os.path.basename(full_path)
    ext = _get_ext(name)
    size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0

    entry = {
        'path': rel,
        'name': name,
        'extension': ext or '(none)',
        'size_bytes': size,
        'format': None,
        'data_type': None,
        'shape_or_rows': None,
        'clinical_ready': None,
        'columns_sample': None,
        'error': None,
    }

    # CSV
    if ext == '.csv' and pd is not None:
        try:
            df_head = pd.read_csv(full_path, nrows=0)
            cols = list(df_head.columns)
            row_count = sum(1 for _ in open(full_path, 'r', encoding='utf-8', errors='replace')) - 1  # minus header
            entry['format'] = 'csv'
            entry['shape_or_rows'] = f'{row_count} rows, {len(cols)} columns'
            entry['row_count'] = row_count
            entry['column_count'] = len(cols)
            entry['data_type'] = infer_csv_data_type(rel, cols)
            entry['columns_sample'] = cols[:20] if len(cols) > 20 else cols
            present, missing = check_clinical_readiness(cols)
            entry['clinical_ready'] = len(missing) == 0
            entry['clinical_columns_present'] = present
            entry['clinical_columns_missing'] = missing
            if entry['data_type'] == 'clinical' and missing:
                entry['label_or_stage'] = (
                    LABEL_COLUMN in [c.lower() for c in cols] or
                    ALTERNATIVE_LABEL_COLUMN in [c.lower() for c in cols] or
                    STAGE_COLUMN in [c.lower() for c in cols]
                )
        except Exception as e:
            entry['format'] = 'csv'
            entry['error'] = str(e)
        results.append(entry)
        return entry.get('data_type')

    # NumPy
    if ext == '.npy' and np is not None:
        try:
            arr = np.load(full_path, allow_pickle=False)
            shape = list(arr.shape)
            entry['format'] = 'numpy'
            entry['shape_or_rows'] = shape
            entry['data_type'] = infer_npy_modality(rel, shape)
            entry['dtype'] = str(arr.dtype)
            results.append(entry)
            return entry['data_type']
        except Exception as e:
            entry['format'] = 'numpy'
            entry['error'] = str(e)
        results.append(entry)
        return None

    # JSON
    if ext == '.json':
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                raw = json.load(f)
            if isinstance(raw, list):
                entry['format'] = 'json'
                entry['shape_or_rows'] = f'list[{len(raw)}]'
                if raw and isinstance(raw[0], dict):
                    entry['columns_sample'] = list(raw[0].keys())[:15]
                    entry['data_type'] = infer_csv_data_type(rel, entry['columns_sample'] or [])
                else:
                    entry['data_type'] = 'unknown'
            elif isinstance(raw, dict):
                entry['format'] = 'json'
                entry['shape_or_rows'] = f'dict[{len(raw)} keys]'
                entry['columns_sample'] = list(raw.keys())[:15]
                entry['data_type'] = 'unknown'
            else:
                entry['format'] = 'json'
                entry['data_type'] = 'unknown'
            results.append(entry)
            return entry.get('data_type')
        except Exception as e:
            entry['format'] = 'json'
            entry['error'] = str(e)
        results.append(entry)
        return None

    # Tabular / other
    if ext in ('.xlsx', '.xls', '.parquet', '.feather'):
        entry['format'] = ext[1:]
        entry['data_type'] = 'tabular'
        if pd is not None and ext == '.parquet':
            try:
                df = pd.read_parquet(full_path, engine='auto')
                entry['shape_or_rows'] = f'{len(df)} rows, {len(df.columns)} cols'
                entry['columns_sample'] = list(df.columns)[:15]
                entry['data_type'] = infer_csv_data_type(rel, list(df.columns))
            except Exception as e:
                entry['error'] = str(e)
        results.append(entry)
        return entry.get('data_type')

    # Image (for future: imaging list or paths) — .nii.gz handled by _get_ext
    if ext in ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.dcm', '.nii', '.nii.gz', '.bmp', '.tiff', '.tif', '.jp2', '.j2k'):
        entry['format'] = 'image'
        entry['data_type'] = 'imaging'
        entry['shape_or_rows'] = 'binary'
        results.append(entry)
        return 'imaging'

    # Text / PDF (metadata or notes)
    if ext in ('.txt', '.pdf', '.md'):
        entry['format'] = ext[1:]
        entry['data_type'] = 'document'
        entry['shape_or_rows'] = 'text'
        results.append(entry)
        return 'document'

    # ZIP (container; we only note it)
    if ext == '.zip':
        entry['format'] = 'archive'
        entry['data_type'] = 'archive'
        entry['shape_or_rows'] = 'container'
        results.append(entry)
        return 'archive'

    entry['data_type'] = 'unknown'
    entry['format'] = ext[1:] if ext else 'unknown'
    results.append(entry)
    return 'unknown'


def scan_directory(root_dir, exclude_dirs=None):
    """Recursively scan root_dir; return (file_entries, folder_tree)."""
    exclude_dirs = set(exclude_dirs or ['.git', '__pycache__', 'node_modules', '.venv', 'venv'])
    root_dir = os.path.abspath(root_dir)
    results = []
    tree = defaultdict(lambda: {'files': [], 'subdirs': [], 'data_type_counts': defaultdict(int)})

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        rel_dir = os.path.relpath(dirpath, root_dir)
        if rel_dir == '.':
            rel_dir = ''
        for f in filenames:
            full = os.path.join(dirpath, f)
            rel_path = os.path.join(rel_dir, f) if rel_dir else f
            dtype = analyze_file(root_dir, rel_path, full, results)
            tree[rel_dir]['files'].append(f)
            if dtype:
                tree[rel_dir]['data_type_counts'][dtype] += 1
        for d in dirnames:
            subrel = os.path.join(rel_dir, d) if rel_dir else d
            tree[rel_dir]['subdirs'].append(subrel)

    return results, dict(tree)


def build_manifest(file_entries, root_dir):
    """Build a manifest for the training system: what data types exist and where."""
    by_type = defaultdict(list)
    clinical_ready = []
    for e in file_entries:
        if e.get('error'):
            continue
        dt = e.get('data_type') or 'unknown'
        by_type[dt].append({
            'path': e['path'],
            'format': e.get('format'),
            'clinical_ready': e.get('clinical_ready'),
            'shape_or_rows': e.get('shape_or_rows'),
            'columns_sample': e.get('columns_sample'),
        })
        if dt == 'clinical' and e.get('clinical_ready'):
            clinical_ready.append(e['path'])

    manifest = {
        'root': os.path.abspath(root_dir),
        'data_types_available': list(by_type.keys()),
        'by_data_type': {k: v for k, v in by_type.items()},
        'clinical_ready_files': clinical_ready,
        'supported_for_training': {
            'clinical': [e['path'] for e in file_entries if e.get('data_type') == 'clinical'],
            'ultrasound': [e['path'] for e in file_entries if e.get('data_type') == 'ultrasound'],
            'genomic': [e['path'] for e in file_entries if e.get('data_type') == 'genomic'],
            'pathology': [e['path'] for e in file_entries if e.get('data_type') == 'pathology'],
            'sensor': [e['path'] for e in file_entries if e.get('data_type') == 'sensor'],
        },
        'canonical_clinical_columns': CLINICAL_FEATURE_COLUMNS,
        'label_columns': [LABEL_COLUMN, ALTERNATIVE_LABEL_COLUMN],
        'stage_column': STAGE_COLUMN,
    }
    return manifest


def load_manifest(manifest_path=None, dataset_root=None):
    """
    Load a previously saved manifest, or scan dataset_root to build one.
    Returns the manifest dict so the training system knows what data types exist.
    """
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    root = dataset_root or (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset'))
    if not os.path.isdir(root):
        return {'root': root, 'data_types_available': [], 'supported_for_training': defaultdict(list)}
    file_entries, _ = scan_directory(root)
    return build_manifest(file_entries, root)


def get_supported_data_types(manifest_path=None, dataset_root=None):
    """
    Return a simple dict of modality -> list of file paths for training.
    Use this so the system knows what types of data can be given.
    """
    manifest = load_manifest(manifest_path=manifest_path, dataset_root=dataset_root)
    root = manifest.get('root', '')
    supported = manifest.get('supported_for_training', {})
    return {
        'clinical': supported.get('clinical', []),
        'ultrasound': supported.get('ultrasound', []),
        'genomic': supported.get('genomic', []),
        'pathology': supported.get('pathology', []),
        'sensor': supported.get('sensor', []),
        'clinical_ready': manifest.get('clinical_ready_files', []),
        'root': root,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze training dataset folders and file formats.')
    parser.add_argument('--root', default=None, help='Dataset root (default: dataset/ next to this script)')
    parser.add_argument('--out', default=None, help='Write manifest to this JSON file')
    parser.add_argument('--quiet', action='store_true', help='Only print summary and errors')
    args = parser.parse_args()

    if args.root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(os.path.dirname(script_dir), 'dataset')
    else:
        root_dir = args.root

    if not os.path.isdir(root_dir):
        print(f"Error: Not a directory: {root_dir}")
        sys.exit(1)

    print(f"Scanning: {root_dir}")
    file_entries, tree = scan_directory(root_dir)
    manifest = build_manifest(file_entries, root_dir)

    # Summary
    print("\n" + "=" * 60)
    print("DATA TYPES AVAILABLE FOR TRAINING")
    print("=" * 60)
    for dt in manifest['data_types_available']:
        paths = manifest['by_data_type'][dt]
        print(f"  {dt}: {len(paths)} file(s)")
        if not args.quiet and len(paths) <= 10:
            for p in paths:
                print(f"    - {p.get('path', p) if isinstance(p, dict) else p}")
        elif not args.quiet:
            for p in paths[:5]:
                print(f"    - {p.get('path', p) if isinstance(p, dict) else p}")
            print(f"    ... and {len(paths) - 5} more")

    print("\n" + "-" * 60)
    print("SUPPORTED MODALITIES (used by FedPINN)")
    print("-" * 60)
    for mod, paths in manifest['supported_for_training'].items():
        print(f"  {mod}: {len(paths)} file(s)")

    print("\n" + "-" * 60)
    print("CLINICAL-READY CSVs (all required columns present)")
    print("-" * 60)
    for p in manifest['clinical_ready_files']:
        print(f"  {p}")
    if not manifest['clinical_ready_files']:
        print("  (none)")

    if not args.quiet:
        print("\n" + "=" * 60)
        print("PER-FILE DETAILS")
        print("=" * 60)
        for e in file_entries:
            print(f"\n  {e['path']}")
            print(f"    format: {e.get('format')}, data_type: {e.get('data_type')}, shape/rows: {e.get('shape_or_rows')}")
            if e.get('clinical_ready') is not None:
                print(f"    clinical_ready: {e['clinical_ready']}, missing: {e.get('clinical_columns_missing', [])}")
            if e.get('error'):
                print(f"    error: {e['error']}")

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written to: {args.out}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
