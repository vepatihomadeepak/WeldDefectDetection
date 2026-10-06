"""
Inference script for weld defect detection.

Supports single image, batch folder, and video inference.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Weld defect detection inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (best.pt)')
    parser.add_argument('--source', type=str, required=True, help='Input image, folder, or video path')
    parser.add_argument('--output', type=str, default='results/predictions', help='Output directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='', help='Device (cuda, cpu)')
    parser.add_argument('--save_txt', action='store_true', help='Save results to YOLO format txt files')
    parser.add_argument('--save_conf', action='store_true', help='Save confidence scores in txt')
    parser.add_argument('--save_crop', action='store_true', help='Save cropped detection boxes')
    parser.add_argument('--show', action='store_true', help='Show results in window')
    parser.add_argument('--line_width', type=int, default=2, help='Bounding box line thickness')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16)')
    return parser.parse_args()


def get_class_names(model):
    """Extract class names from model."""
    if hasattr(model, 'names'):
        return model.names
    return {i: f'class_{i}' for i in range(100)}


def process_image(model, image_path: Path, output_dir: Path, args, class_names: dict) -> dict:
    """Process a single image and return detection results."""
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        verbose=False,
        save=False,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        line_width=args.line_width
    )
    
    result = results[0]
    detections = []
    
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            detections.append({
                'class_id': cls_id,
                'class_name': class_names.get(cls_id, f'class_{cls_id}'),
                'confidence': conf,
                'bbox': xyxy.tolist()  # [x1, y1, x2, y2]
            })
    
    # Save annotated image
    annotated = result.plot(line_width=args.line_width)
    output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated)
    
    # Save YOLO format labels if requested
    if args.save_txt:
        label_path = output_dir / 'labels' / f"{image_path.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                h, w = annotated.shape[:2]
                x_center = (x1 + x2) / 2.0 / w
                y_center = (y1 + y2) / 2.0 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                line = f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                if args.save_conf:
                    line += f" {det['confidence']:.6f}"
                f.write(line + '\n')
    
    return {
        'image': str(image_path),
        'detections': detections,
        'output_image': str(output_path),
        'num_detections': len(detections)
    }


def process_folder(model, folder_path: Path, output_dir: Path, args, class_names: dict) -> list:
    """Process all images in a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return []
    
    print(f"Processing {len(image_files)} images...")
    results = []
    
    for img_path in image_files:
        try:
            result = process_image(model, img_path, output_dir, args, class_names)
            results.append(result)
            print(f"  {img_path.name}: {result['num_detections']} detections")
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    return results


def process_video(model, video_path: Path, output_dir: Path, args, class_names: dict) -> dict:
    """Process video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = output_dir / f"{video_path.stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    print(f"Processing video: {video_path.name} ({total_frames} frames)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            verbose=False
        )
        
        annotated = results[0].plot(line_width=args.line_width)
        writer.write(annotated)
        
        if results[0].boxes is not None:
            total_detections += len(results[0].boxes)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    writer.release()
    
    result = {
        'video': str(video_path),
        'output_video': str(output_path),
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / max(frame_count, 1)
    }
    
    print(f"Video processing complete. Output: {output_path}")
    return result


def main():
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source not found: {source_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if args.device else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Weld Defect Detection - Inference")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Source: {source_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Confidence: {args.conf}, IoU: {args.iou}")
    print("=" * 60)
    
    model = YOLO(str(model_path))
    class_names = get_class_names(model)
    
    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}:
            result = process_image(model, source_path, output_dir, args, class_names)
            print(f"\nDetections: {result['num_detections']}")
            for det in result['detections']:
                print(f"  {det['class_name']}: {det['confidence']:.2%} at {det['bbox']}")
            print(f"Output saved to: {result['output_image']}")
        
        elif suffix in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
            result = process_video(model, source_path, output_dir, args, class_names)
            print(f"\nFrames processed: {result['frames_processed']}")
            print(f"Total detections: {result['total_detections']}")
            print(f"Output saved to: {result['output_video']}")
        
        else:
            print(f"Unsupported file format: {suffix}")
            sys.exit(1)
    
    elif source_path.is_dir():
        results = process_folder(model, source_path, output_dir, args, class_names)
        total_dets = sum(r['num_detections'] for r in results)
        print(f"\nTotal images: {len(results)}")
        print(f"Total detections: {total_dets}")
    
    else:
        print(f"Source is not a file or directory: {source_path}")
        sys.exit(1)
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()