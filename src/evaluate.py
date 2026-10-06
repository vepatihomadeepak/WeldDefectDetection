"""
Evaluation script for weld defect detection model.

Computes precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95,
and generates confusion matrix and visualization plots.
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 weld defect detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (best.pt)')
    parser.add_argument('--data', type=str, default='data.yaml', help='Dataset configuration YAML')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='', help='Device (cuda, cpu)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for NMS')
    parser.add_argument('--save_json', action='store_true', help='Save results to JSON')
    parser.add_argument('--save_plots', action='store_true', default=True, help='Save visualization plots')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--plots_dir', type=str, default='results/plots', help='Directory for plots')
    return parser.parse_args()


def evaluate_model(model_path: str, data_yaml: str, split: str, imgsz: int, batch: int, 
                   device: str, conf: float, iou: float):
    """Run evaluation and return metrics."""
    model = YOLO(model_path)
    
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {data_yaml} (split: {split})")
    print(f"Confidence threshold: {conf}, IoU threshold: {iou}")
    
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        verbose=True
    )
    
    return results


def extract_metrics(results, class_names: list) -> dict:
    """Extract key metrics from validation results."""
    metrics = {}
    
    # Overall metrics
    metrics['precision'] = float(results.box.mp)
    metrics['recall'] = float(results.box.mr)
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-16)
    metrics['map50'] = float(results.box.map50)
    metrics['map50_95'] = float(results.box.map)
    
    # Per-class metrics
    metrics['per_class'] = {}
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
            'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0,
            'f1': 0.0,
            'map50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0.0,
            'map50_95': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0,
            'support': int(results.box.nt_per_class[i]) if hasattr(results.box, 'nt_per_class') and i < len(results.box.nt_per_class) else 0
        }
        p = metrics['per_class'][name]['precision']
        r = metrics['per_class'][name]['recall']
        metrics['per_class'][name]['f1'] = 2 * p * r / (p + r + 1e-16)
    
    # Confusion matrix
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        metrics['confusion_matrix'] = results.confusion_matrix.matrix.tolist()
    else:
        metrics['confusion_matrix'] = None
    
    return metrics


def save_metrics_json(metrics: dict, output_path: Path):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plot and save confusion matrix."""
    if cm is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Convert to int if float
    cm_int = cm.astype(int)
    sns.heatmap(cm_int, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names + ['background'],
                yticklabels=class_names + ['background'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_per_class_metrics(metrics: dict, class_names: list, output_path: Path):
    """Plot per-class precision, recall, F1, mAP."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    classes = class_names
    x = np.arange(len(classes))
    width = 0.2
    
    precision = [metrics['per_class'][c]['precision'] for c in classes]
    recall = [metrics['per_class'][c]['recall'] for c in classes]
    f1 = [metrics['per_class'][c]['f1'] for c in classes]
    map50 = [metrics['per_class'][c]['map50'] for c in classes]
    map50_95 = [metrics['per_class'][c]['map50_95'] for c in classes]
    
    # Precision, Recall, F1
    axes[0, 0].bar(x - width, precision, width, label='Precision', color='skyblue')
    axes[0, 0].bar(x, recall, width, label='Recall', color='lightgreen')
    axes[0, 0].bar(x + width, f1, width, label='F1', color='salmon')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Per-Class Precision, Recall, F1')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.05)
    
    # mAP@0.5 and mAP@0.5:0.95
    axes[0, 1].bar(x - width/2, map50, width, label='mAP@0.5', color='orange')
    axes[0, 1].bar(x + width/2, map50_95, width, label='mAP@0.5:0.95', color='purple')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].set_title('Per-Class mAP')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.05)
    
    # Support (number of instances)
    support = [metrics['per_class'][c]['support'] for c in classes]
    axes[1, 0].bar(x, support, color='gray')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Per-Class Instance Count (Support)')
    
    # Overall metrics summary
    axes[1, 1].axis('off')
    summary_text = f"""Overall Metrics:
    
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1']:.4f}
mAP@0.5:   {metrics['map50']:.4f}
mAP@0.5:0.95: {metrics['map50_95']:.4f}
"""
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-class metrics plot saved to: {output_path}")


def main():
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_path}")
        sys.exit(1)
    
    import yaml
    with open(data_path) as f:
        data_config = yaml.safe_load(f)
    names = data_config.get('names', {})
    # Convert dict to list if needed
    if isinstance(names, dict):
        class_names = [names.get(i, f'class_{i}') for i in range(len(names))]
    else:
        class_names = names
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Weld Defect Detection - Evaluation")
    print("=" * 60)
    
    device = args.device if args.device else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    results = evaluate_model(
        str(model_path), str(data_path), args.split,
        args.imgsz, args.batch, device, args.conf, args.iou
    )
    
    metrics = extract_metrics(results, class_names)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1']:.4f}")
    print(f"mAP@0.5:        {metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95:   {metrics['map50_95']:.4f}")
    print()
    print("Per-class metrics:")
    for cls_name, cls_metrics in metrics['per_class'].items():
        name_str = str(cls_name)
        print(f"  {name_str:20s} P={cls_metrics['precision']:.3f} R={cls_metrics['recall']:.3f} "
              f"F1={cls_metrics['f1']:.3f} mAP50={cls_metrics['map50']:.3f} mAP50-95={cls_metrics['map50_95']:.3f}")
    print("=" * 60)
    
    # Save metrics JSON
    if args.save_json:
        json_path = output_dir / f'evaluation_{args.split}_metrics.json'
        save_metrics_json(metrics, json_path)
    
    # Save plots
    if args.save_plots:
        if metrics['confusion_matrix'] is not None:
            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(cm, class_names, plots_dir / f'confusion_matrix_{args.split}.png')
        
        plot_per_class_metrics(metrics, class_names, plots_dir / f'per_class_metrics_{args.split}.png')
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()