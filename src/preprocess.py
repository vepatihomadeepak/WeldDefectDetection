"""
Preprocessing module for weld defect detection.

Provides image preprocessing and augmentation utilities for weld images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    """Get training augmentation pipeline."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.2, 0.44), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    """Get validation/test transforms (no augmentation)."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_inference_transforms(img_size: int = 640) -> A.Compose:
    """Get inference transforms for single images."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def enhance_weld_image(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Apply weld-specific image enhancement.
    
    Args:
        image: Input image (BGR format)
        method: Enhancement method ('clahe', 'sharpen', 'denoise', 'all')
    
    Returns:
        Enhanced image
    """
    enhanced = image.copy()
    
    if method in ('clahe', 'all'):
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if method in ('sharpen', 'all'):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    if method in ('denoise', 'all'):
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced


def validate_dataset_structure(data_root: Path) -> dict:
    """
    Validate dataset structure and return statistics.
    
    Args:
        data_root: Root directory containing train/val/test subdirectories
    
    Returns:
        Dictionary with validation results and statistics
    """
    splits = ['train', 'val', 'test']
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    for split in splits:
        img_dir = data_root / split / 'images'
        lbl_dir = data_root / split / 'labels'
        
        if not img_dir.exists():
            results['errors'].append(f"Missing directory: {img_dir}")
            results['valid'] = False
            continue
        
        if not lbl_dir.exists():
            results['warnings'].append(f"Missing labels directory: {lbl_dir}")
        
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        labels = list(lbl_dir.glob('*.txt'))
        
        results['stats'][split] = {
            'images': len(images),
            'labels': len(labels),
            'matched': len(set(img.stem for img in images) & set(lbl.stem for lbl in labels))
        }
        
        if len(images) == 0:
            results['warnings'].append(f"No images found in {split}")
    
    return results


def convert_voc_to_yolo(voc_root: Path, output_root: Path, class_names: List[str]) -> None:
    """
    Convert Pascal VOC XML annotations to YOLO format.
    
    Args:
        voc_root: Root directory with Annotations/ and JPEGImages/
        output_root: Output directory for YOLO format (images/ and labels/)
        class_names: List of class names in order
    """
    import xml.etree.ElementTree as ET
    
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    annotations_dir = voc_root / 'Annotations'
    images_dir = voc_root / 'JPEGImages'
    
    for split in ['train', 'val', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Simple 70/15/15 split
    xml_files = list(annotations_dir.glob('*.xml'))
    np.random.shuffle(xml_files)
    
    n_total = len(xml_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    splits = {
        'train': xml_files[:n_train],
        'val': xml_files[n_train:n_train + n_val],
        'test': xml_files[n_train + n_val:]
    }
    
    for split_name, files in splits.items():
        for xml_file in files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
            img_name = root.find('filename').text
            src_img = images_dir / img_name
            dst_img = output_root / split_name / 'images' / img_name
            
            if src_img.exists():
                import shutil
                shutil.copy2(src_img, dst_img)
            
            label_lines = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in class_to_idx:
                    continue
                cls_idx = class_to_idx[cls_name]
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = (xmin + xmax) / 2.0 / img_w
                y_center = (ymin + ymax) / 2.0 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h
                
                label_lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            label_file = output_root / split_name / 'labels' / f"{Path(img_name).stem}.txt"
            label_file.write_text('\n'.join(label_lines))


def visualize_annotations(image_path: Path, label_path: Path, class_names: List[str], 
                         output_path: Optional[Path] = None) -> np.ndarray:
    """
    Draw bounding boxes on image for visualization.
    
    Args:
        image_path: Path to image file
        label_path: Path to YOLO format label file
        class_names: List of class names
        output_path: Optional path to save visualization
    
    Returns:
        Annotated image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_idx, x_center, y_center, width, height = map(float, parts)
                cls_idx = int(cls_idx)
                
                # Convert from normalized to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(str(output_path), image)
    
    return image


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
        results = validate_dataset_structure(data_root)
        print(f"Dataset valid: {results['valid']}")
        print(f"Stats: {results['stats']}")
        if results['errors']:
            print(f"Errors: {results['errors']}")
        if results['warnings']:
            print(f"Warnings: {results['warnings']}")
    else:
        print("Usage: python preprocess.py <data_root>")