#!/usr/bin/env python3
"""
Resume finetuning from the v2 checkpoint with smaller batch to avoid OOM crash.
Continues from where the previous run left off.
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()

# Resume from v2's best checkpoint
BASE_MODEL = PROJECT_ROOT / "runs" / "detect" / "weld_defect_finetune_v2" / "weights" / "best.pt"
DATA_CONFIG = PROJECT_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "weld_defect_finetune_v3"

# Where the app expects the model
APP_MODEL = PROJECT_ROOT / "models" / "best.pt"
# Original best model for comparison
ORIGINAL_BEST = PROJECT_ROOT / "runs" / "detect" / "weld_defect" / "weights" / "best.pt"


def main():
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")

    print("=" * 60)
    print("Weld Defect Detection - Finetuning v3 (resumed)")
    print("=" * 60)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Data config: {DATA_CONFIG}")
    print()

    model = YOLO(str(BASE_MODEL))
    print(f"Model classes: {model.names}")

    results = model.train(
        data=str(DATA_CONFIG),
        epochs=80,
        patience=20,
        batch=8,                  # Smaller batch to avoid CUDA OOM
        imgsz=640,
        device="0",
        workers=4,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,

        # Optimizer - even lower LR since we're further into finetuning
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Higher classification loss for better defect discrimination
        box=7.5,
        cls=2.0,                 # Even higher cls weight
        dfl=1.5,

        # Strong augmentation
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.3,

        dropout=0.1,
        close_mosaic=15,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        seed=42,
        amp=True,
    )

    # Copy best weights if they beat the original
    best_weights = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    if best_weights.is_file():
        # Check if new model is better than original
        new_model = YOLO(str(best_weights))
        new_val = new_model.val(data=str(DATA_CONFIG), split="val")
        new_map50 = new_val.box.map50

        original_model = YOLO(str(ORIGINAL_BEST))
        orig_val = original_model.val(data=str(DATA_CONFIG), split="val")
        orig_map50 = orig_val.box.map50

        print(f"\nOriginal mAP50: {orig_map50:.4f}")
        print(f"New mAP50:      {new_map50:.4f}")

        if new_map50 > orig_map50:
            shutil.copy2(best_weights, APP_MODEL)
            print(f"NEW MODEL IS BETTER! Copied to {APP_MODEL}")
        else:
            # Still copy original to ensure app works
            shutil.copy2(ORIGINAL_BEST, APP_MODEL)
            print(f"Original model still better. Keeping original at {APP_MODEL}")

    print("\nFinetuning v3 complete!")


if __name__ == "__main__":
    main()
