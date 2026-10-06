"""
Convert YOLO bounding box labels to polygon format for YOLOv8-seg.

YOLO bbox format: class_id x_center y_center width height (normalized)
YOLO seg format: class_id x1 y1 x2 y2 x3 y3 x4 y4 ... (normalized polygon points)
"""

import argparse
from pathlib import Path
import numpy as np


def bbox_to_polygon(cls_id: int, x_center: float, y_center: float, width: float, height: float) -> list:
    """
    Convert YOLO bbox to 4-point rectangle polygon.
    Returns list of normalized coordinates: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center - height / 2
    x3 = x_center + width / 2
    y3 = y_center + height / 2
    x4 = x_center - width / 2
    y4 = y_center + height / 2
    
    # Clamp to [0, 1]
    coords = [x1, y1, x2, y2, x3, y3, x4, y4]
    coords = [max(0.0, min(1.0, c)) for c in coords]
    return coords


def convert_labels(input_dir: Path, output_dir: Path):
    """Convert all label files in input_dir to polygon format in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(input_dir.glob('*.txt'))
    print(f"Converting {len(label_files)} label files...")
    
    converted = 0
    for label_file in label_files:
        content = label_file.read_text().strip()
        if not content:
            # Empty label file - copy as-is
            (output_dir / label_file.name).write_text('')
            continue
        
        lines = content.split('\n')
        output_lines = []
        
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to polygon
            polygon = bbox_to_polygon(cls_id, x_center, y_center, width, height)
            poly_str = ' '.join(f'{c:.6f}' for c in polygon)
            output_lines.append(f'{cls_id} {poly_str}')
        
        (output_dir / label_file.name).write_text('\n'.join(output_lines))
        converted += 1
    
    print(f"Converted {converted} files to {output_dir}")
    return converted


def convert_dataset_splits(data_root: Path):
    """Convert train/val/test label directories."""
    for split in ['train', 'val', 'test']:
        in_dir = data_root / split / 'labels'
        out_dir = data_root / split / 'labels_seg'
        
        if in_dir.exists():
            print(f"\n--- {split} ---")
            convert_labels(in_dir, out_dir)
        else:
            print(f"Skipping {split}: {in_dir} not found")


def verify_conversion(data_root: Path, split: str = 'train'):
    """Verify converted labels have correct format."""
    seg_dir = data_root / split / 'labels_seg'
    if not seg_dir.exists():
        print(f"Segmentation labels not found: {seg_dir}")
        return
    
    for f in list(seg_dir.glob('*.txt'))[:3]:
        content = f.read_text().strip()
        if content:
            parts = content.split()
            print(f"{f.name}: {len(parts)} values per line (class_id + {len(parts)-1} coords)")
            if (len(parts) - 1) % 2 == 0:
                print(f"  -> Valid polygon with {(len(parts)-1)//2} points")
            else:
                print(f"  -> WARNING: Odd number of coordinates!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert YOLO bbox labels to polygon format for segmentation')
    parser.add_argument('--data-root', type=str, default='data', help='Dataset root directory')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing conversion')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if args.verify_only:
        for split in ['train', 'val', 'test']:
            verify_conversion(data_root, split)
    else:
        convert_dataset_splits(data_root)
        print("\n--- Verification ---")
        for split in ['train', 'val', 'test']:
            verify_conversion(data_root, split)