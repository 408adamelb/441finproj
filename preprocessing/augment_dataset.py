import csv
import random
from pathlib import Path
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CLASSES = ["stop_sign", "yield_sign", "speed_limit_sign", "apple", "banana", "orange"]
TRAIN_CSV = Path("data/splits/train.csv")
PROCESSED_DIR = Path("data/processed")

NUM_AUGMENTATIONS = 3


def get_augmentation_transform():
    return T.Compose([
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])


def apply_gaussian_blur(img, p=0.2):
    if random.random() < p:
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def augment_image(img_path, output_dir, num_augmentations=3):
    img = Image.open(img_path).convert("RGB")
    stem = img_path.stem
    
    transform = get_augmentation_transform()
    
    augmented_paths = []
    for i in range(1, num_augmentations + 1):
        aug_img = transform(img)
        aug_img = apply_gaussian_blur(aug_img, p=0.2)
        aug_filename = f"{stem}_aug{i}.png"
        aug_path = output_dir / aug_filename
        aug_img.save(aug_path)
        augmented_paths.append(aug_path)
    
    return augmented_paths


def augment_training_set():
    print("=" * 60)
    print("Data Augmentation")
    print(f"Training CSV: {TRAIN_CSV}")
    print(f"Augmentations per image: {NUM_AUGMENTATIONS}")
    print("=" * 60)
    
    train_files = {}
    for class_name in CLASSES:
        train_files[class_name] = []
    
    with open(TRAIN_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = Path(row["filepath"])
            label = row["label"]
            if "_aug" not in filepath.stem:  # skip already augmented
                train_files[label].append(filepath)
    
    total_original = sum(len(files) for files in train_files.values())
    print(f"\nOriginal training images: {total_original}")
    print(f"Will generate: {total_original * NUM_AUGMENTATIONS} augmented images")
    
    results = {}
    total_augmented = 0
    
    for class_name in CLASSES:
        files = train_files[class_name]
        print(f"\n[{class_name}]")
        print(f"  Original images: {len(files)}")
        
        output_dir = PROCESSED_DIR / class_name
        augmented_count = 0
        
        for i, img_path in enumerate(files):
            if not img_path.exists():
                print(f"  Warning: {img_path} not found, skipping")
                continue
            
            aug_paths = augment_image(img_path, output_dir, NUM_AUGMENTATIONS)
            augmented_count += len(aug_paths)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(files)} images...")
        
        results[class_name] = {
            "original": len(files),
            "augmented": augmented_count
        }
        total_augmented += augmented_count
        print(f"  Generated: {augmented_count} augmented images")
    
    # Print summary
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Original':<12} {'Augmented':<12} {'Total':<12}")
    print("-" * 56)
    
    for class_name, stats in results.items():
        total = stats['original'] + stats['augmented']
        print(f"{class_name:<20} {stats['original']:<12} {stats['augmented']:<12} {total:<12}")
    
    print("-" * 56)
    print(f"{'TOTAL':<20} {total_original:<12} {total_augmented:<12} {total_original + total_augmented:<12}")
    print("=" * 60)
    
    print("\nUpdating train.csv with augmented images...")
    new_train_rows = []
    
    with open(TRAIN_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_train_rows.append(row)
            filepath = Path(row["filepath"])
            if "_aug" not in filepath.stem:
                for i in range(1, NUM_AUGMENTATIONS + 1):
                    aug_filename = f"{filepath.stem}_aug{i}.png"
                    aug_path = filepath.parent / aug_filename
                    if aug_path.exists():
                        new_train_rows.append({
                            "filepath": str(aug_path),
                            "label": row["label"]
                        })
    
    with open(TRAIN_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label"])
        writer.writeheader()
        writer.writerows(new_train_rows)
    
    print(f"Updated train.csv: {len(new_train_rows)} total entries")
    print("\nDone! You can now retrain with augmented data.")


if __name__ == "__main__":
    random.seed(42)
    augment_training_set()

