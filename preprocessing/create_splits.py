import csv
import random
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

CLASSES = ["stop_sign", "yield_sign", "speed_limit_sign", "apple", "banana", "orange"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_splits(seed=42):
    random.seed(seed)
    
    print("=" * 60)
    print("Creating Stratified Train/Val/Test Splits")
    print(f"Source: {PROCESSED_DIR}")
    print(f"Target: {SPLITS_DIR}")
    print(f"Ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"Random seed: {seed}")
    print("=" * 60)
    
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_data = []
    val_data = []
    test_data = []
    
    results = {}
    
    for class_name in CLASSES:
        input_dir = PROCESSED_DIR / class_name
        
        print(f"\n[{class_name}]")
        
        if not input_dir.exists():
            print(f"  Skipping - directory not found: {input_dir}")
            continue
        
        image_files = list(input_dir.glob("*.png"))
        random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        for f in train_files:
            train_data.append({"filepath": str(f), "label": class_name})
        for f in val_files:
            val_data.append({"filepath": str(f), "label": class_name})
        for f in test_files:
            test_data.append({"filepath": str(f), "label": class_name})
        
        results[class_name] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": total
        }
        
        print(f"  Total: {total}")
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    def save_csv(data, filename):
        filepath = SPLITS_DIR / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
            writer.writeheader()
            writer.writerows(data)
        print(f"  Saved: {filepath} ({len(data)} rows)")
    
    print("\nSaving CSV files...")
    save_csv(train_data, "train.csv")
    save_csv(val_data, "val.csv")
    save_csv(test_data, "test.csv")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    total_train = total_val = total_test = total_all = 0
    
    for class_name, stats in results.items():
        print(f"{class_name:<20} {stats['train']:<10} {stats['val']:<10} {stats['test']:<10} {stats['total']:<10}")
        total_train += stats['train']
        total_val += stats['val']
        total_test += stats['test']
        total_all += stats['total']
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
    print("=" * 60)


if __name__ == "__main__":
    create_splits()
