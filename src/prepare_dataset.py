"""
Dataset preparation utilities for weld defect detection.

Helps users download, organize, and validate datasets.
"""

import argparse
import sys
from pathlib import Path
import shutil
import yaml
import numpy as np


def create_sample_dataset_structure(data_root: Path):
    """Create empty dataset folder structure."""
    splits = ['train', 'val', 'test']
    for split in splits:
        (data_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Created dataset structure at: {data_root}")
    print("Place your images in <split>/images/ and YOLO labels in <split>/labels/")


def validate_yolo_labels(label_dir: Path, num_classes: int) -> dict:
    """Validate YOLO format label files."""
    results = {
        'valid': True,
        'total_files': 0,
        'total_boxes': 0,
        'class_counts': {},
        'errors': [],
        'warnings': []
    }
    
    label_files = list(label_dir.glob('*.txt'))
    results['total_files'] = len(label_files)
    
    for label_file in label_files:
        try:
            with open(label_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        results['errors'].append(f"{label_file.name}:{line_num}: Invalid format (need 5+ values)")
                        results['valid'] = False
                        continue
                    
                    try:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                    except ValueError:
                        results['errors'].append(f"{label_file.name}:{line_num}: Non-numeric values")
                        results['valid'] = False
                        continue
                    
                    if cls_id < 0 or cls_id >= num_classes:
                        results['errors'].append(f"{label_file.name}:{line_num}: Class ID {cls_id} out of range [0, {num_classes-1}]")
                        results['valid'] = False
                    
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        results['warnings'].append(f"{label_file.name}:{line_num}: Normalized coords out of [0,1] range")
                    
                    results['total_boxes'] += 1
                    results['class_counts'][cls_id] = results['class_counts'].get(cls_id, 0) + 1
        
        except Exception as e:
            results['errors'].append(f"{label_file.name}: {e}")
            results['valid'] = False
    
    return results


def split_dataset(images_dir: Path, labels_dir: Path, output_dir: Path, 
                  train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """Split dataset into train/val/test."""
    np.random.seed(seed)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
        image_files.extend(images_dir.glob(ext))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Match with labels
    paired = []
    for img in image_files:
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            paired.append((img, label))
        else:
            print(f"Warning: No label for {img.name}")
    
    if not paired:
        print("No matched image-label pairs found")
        return
    
    np.random.shuffle(paired)
    n = len(paired)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    
    splits = {
        'train': paired[:n_train],
        'val': paired[n_train:n_train + n_val],
        'test': paired[n_train + n_val:]
    }
    
    for split_name, pairs in splits.items():
        img_out = output_dir / split_name / 'images'
        lbl_out = output_dir / split_name / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img, lbl in pairs:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)
        
        print(f"{split_name}: {len(pairs)} pairs")
    
    print(f"Dataset split complete. Output: {output_dir}")


def generate_data_yaml(data_root: Path, class_names: list, output_path: Path = None):
    """Generate data.yaml for YOLO training."""
    if output_path is None:
        output_path = data_root.parent / 'data.yaml'
    
    config = {
        'path': str(data_root.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated {output_path}")


def download_kaggle_dataset(dataset_slug: str, output_dir: Path):
    """Download dataset from Kaggle (requires kaggle CLI and API token)."""
    import subprocess
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', str(output_dir), '--unzip'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"Downloaded {dataset_slug} to {output_dir}")
        else:
            print(f"Kaggle download failed: {result.stderr}")
            print("Make sure you have kaggle CLI installed and API token configured")
    except FileNotFoundError:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        print("Then configure API token from https://www.kaggle.com/settings/account")
    except subprocess.TimeoutExpired:
        print("Download timed out")


def main():
    parser = argparse.ArgumentParser(description='Dataset preparation utilities')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create structure
    create_parser = subparsers.add_parser('create-structure', help='Create empty dataset folder structure')
    create_parser.add_argument('--data-root', type=str, default='data', help='Dataset root directory')
    
    # Validate
    validate_parser = subparsers.add_parser('validate', help='Validate YOLO label files')
    validate_parser.add_argument('--labels-dir', type=str, required=True, help='Labels directory')
    validate_parser.add_argument('--num-classes', type=int, default=8, help='Number of classes')
    
    # Split
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--images-dir', type=str, required=True, help='Images directory')
    split_parser.add_argument('--labels-dir', type=str, required=True, help='Labels directory')
    split_parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    split_parser.add_argument('--train-ratio', type=float, default=0.7)
    split_parser.add_argument('--val-ratio', type=float, default=0.15)
    split_parser.add_argument('--test-ratio', type=float, default=0.15)
    split_parser.add_argument('--seed', type=int, default=42)
    
    # Generate data.yaml
    yaml_parser = subparsers.add_parser('generate-yaml', help='Generate data.yaml')
    yaml_parser.add_argument('--data-root', type=str, default='data', help='Dataset root')
    yaml_parser.add_argument('--classes', type=str, nargs='+', 
                            default=['crack', 'porosity', 'undercut', 'lack_of_fusion', 
                                    'overlap', 'slag_inclusion', 'burn_through', 'spatter'],
                            help='Class names')
    yaml_parser.add_argument('--output', type=str, help='Output path')
    
    # Download
    download_parser = subparsers.add_parser('download', help='Download from Kaggle')
    download_parser.add_argument('--dataset', type=str, default='sukmaadhiwijaya/welding-defect-object-detection', help='Kaggle dataset slug')
    download_parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'create-structure':
        create_sample_dataset_structure(Path(args.data_root))
    
    elif args.command == 'validate':
        results = validate_yolo_labels(Path(args.labels_dir), args.num_classes)
        print(f"Valid: {results['valid']}")
        print(f"Files: {results['total_files']}")
        print(f"Total boxes: {results['total_boxes']}")
        print(f"Class counts: {results['class_counts']}")
        if results['errors']:
            print("Errors:")
            for e in results['errors']:
                print(f"  - {e}")
        if results['warnings']:
            print("Warnings:")
            for w in results['warnings']:
                print(f"  - {w}")
    
    elif args.command == 'split':
        split_dataset(Path(args.images_dir), Path(args.labels_dir), Path(args.output_dir),
                      args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    elif args.command == 'generate-yaml':
        generate_data_yaml(Path(args.data_root), args.classes, Path(args.output) if args.output else None)
    
    elif args.command == 'download':
        download_kaggle_dataset(args.dataset, Path(args.output_dir))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()