"""Train a YOLO weld defect detector using the project's bounding-box labels.

The bundled labels are detection labels, not true polygon masks, so use a
detection model such as ``yolov8n.pt`` rather than a YOLO ``-seg`` model.
"""

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO weld defect detector')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset configuration YAML')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Detection model to train (for example yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0 for GPU 0, cpu, or cuda)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='weld_defect', help='Experiment name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer (SGD, Adam, AdamW, auto)')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--cache', action='store_true', help='Cache images in RAM')
    parser.add_argument('--rect', action='store_true', help='Rectangular training')
    parser.add_argument('--cos_lr', action='store_true', help='Cosine LR scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Disable mosaic augmentation for last N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--amp', action='store_true', default=True, help='Automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic training')
    return parser.parse_args()


def check_dataset(data_yaml: Path) -> bool:
    """Verify dataset exists and has valid structure."""
    import yaml
    
    if not data_yaml.exists():
        print(f"Error: Dataset config not found: {data_yaml}")
        return False
    
    with open(data_yaml) as f:
        config = yaml.safe_load(f)
    
    data_root = Path(config.get('path', '.'))
    train_dir = data_root / config.get('train', 'train/images')
    val_dir = data_root / config.get('val', 'val/images')
    
    if not train_dir.exists():
        print(f"Error: Training images directory not found: {train_dir}")
        return False
    
    if not val_dir.exists():
        print(f"Error: Validation images directory not found: {val_dir}")
        return False
    
    train_images = list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png')) + list(train_dir.glob('*.jpeg'))
    val_images = list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpeg'))
    
    if len(train_images) == 0:
        print(f"Error: No training images found in {train_dir}")
        return False
    
    if len(val_images) == 0:
        print(f"Error: No validation images found in {val_dir}")
        return False
    
    train_labels = data_root / 'train' / 'labels'
    val_labels = data_root / 'val' / 'labels'
    if not train_labels.exists() or not val_labels.exists():
        print("Error: YOLO detection labels are missing from data/<split>/labels")
        return False
    print(f"Found {len(list(train_labels.glob('*.txt')))} training and {len(list(val_labels.glob('*.txt')))} validation label files")
    
    print(f"Dataset OK: {len(train_images)} train images, {len(val_images)} val images")
    return True


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Weld Defect Detection - Training (YOLO)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    
    data_path = Path(args.data).resolve()
    if not check_dataset(data_path):
        sys.exit(1)
    if '-seg' in Path(args.model).name.lower():
        print("Error: this dataset contains bounding-box labels; use yolov8n.pt (not yolov8n-seg.pt).")
        sys.exit(2)
    project_path = Path(args.project).resolve()
    
    # Device handling - default to GPU 0
    if args.device:
        device = args.device
    else:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    if device != 'cpu' and torch.cuda.is_available():
        gpu_id = int(device) if device.isdigit() else 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    
    model = YOLO(args.model)
    
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        rect=args.rect,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        amp=args.amp,
        seed=args.seed,
        deterministic=args.deterministic,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved under: {project_path}")
    print(f"Last model saved under: {project_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
