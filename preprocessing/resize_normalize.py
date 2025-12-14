import os
from pathlib import Path
from PIL import Image
import numpy as np

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TARGET_SIZE = (64, 64)

CLASSES = ["stop_sign", "yield_sign", "speed_limit_sign", "apple", "banana", "orange"]


def process_image(input_path, output_path):
    try:
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_normalized = Image.fromarray((img_array * 255).astype(np.uint8))
        output_path = output_path.with_suffix(".png")
        img_normalized.save(output_path, "PNG")
        
        return True
    except Exception as e:
        print(f"    [FAILED] {input_path.name}: {e}")
        return False


def preprocess_images():
    print("=" * 60)
    print("Image Preprocessing")
    print(f"Source: {RAW_DIR}")
    print(f"Target: {PROCESSED_DIR}")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("=" * 60)
    
    total_processed = 0
    total_failed = 0
    results = {}
    
    for class_name in CLASSES:
        input_dir = RAW_DIR / class_name
        output_dir = PROCESSED_DIR / class_name
        
        print(f"\n[{class_name}]")
        
        if not input_dir.exists():
            print(f"  Skipping - directory not found: {input_dir}")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        image_files = [f for f in input_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        print(f"  Found {len(image_files)} images")
        
        processed = 0
        failed = 0
        
        for i, img_path in enumerate(image_files):
            output_path = output_dir / img_path.stem
            
            if process_image(img_path, output_path):
                processed += 1
            else:
                failed += 1
            
            if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
                print(f"  Progress: {i + 1}/{len(image_files)} ({processed} processed, {failed} failed)")
        
        results[class_name] = {"processed": processed, "failed": failed}
        total_processed += processed
        total_failed += failed
        
        print(f"  Done: {processed} processed, {failed} failed")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Processed':<12} {'Failed':<10}")
    print("-" * 42)
    
    for class_name, stats in results.items():
        print(f"{class_name:<20} {stats['processed']:<12} {stats['failed']:<10}")
    
    print("-" * 42)
    print(f"{'TOTAL':<20} {total_processed:<12} {total_failed:<10}")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_images()

