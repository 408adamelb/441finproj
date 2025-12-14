import sys
import csv
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent.parent / "models"))
from mlp import MLP
from cnn import CNN


CLASSES = ["stop_sign", "yield_sign", "speed_limit_sign", "apple", "banana", "orange"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class ImageDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filepath": row["filepath"],
                    "label": CLASS_TO_IDX[row["label"]]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = Image.open(sample["filepath"]).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        return img_tensor, sample["label"]


def compute_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def print_confusion_matrix(matrix, class_names):
    header = "Actual \\ Predicted"
    col_width = max(len(c) for c in class_names) + 2
    
    print("\nConfusion Matrix:")
    print("=" * (len(header) + len(class_names) * col_width + 5))
    
    print(f"{'':<{len(header)}}", end="")
    for name in class_names:
        short_name = name[:col_width-1]
        print(f"{short_name:>{col_width}}", end="")
    print()
    
    print("-" * (len(header) + len(class_names) * col_width))
    
    for i, name in enumerate(class_names):
        short_name = name[:len(header)-1]
        print(f"{short_name:<{len(header)}}", end="")
        for j in range(len(class_names)):
            print(f"{matrix[i, j]:>{col_width}}", end="")
        print()
    
    print("=" * (len(header) + len(class_names) * col_width + 5))


def compute_per_class_metrics(matrix, class_names):
    metrics = []
    
    for i, name in enumerate(class_names):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": matrix[i, :].sum()
        })
    
    return metrics


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    print(f"Creating {args.model.upper()} model...")
    if args.model == "mlp":
        model = MLP(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_layers,
            num_classes=len(CLASSES)
        )
    else:
        model = CNN(
            num_conv_layers=args.num_layers,
            base_filters=args.hidden_dim,
            dropout=0,  # No dropout during evaluation
            num_classes=len(CLASSES)
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation accuracy at save: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    print(f"\nLoading test data: {args.test_csv}")
    test_dataset = ImageDataset(args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    confusion = compute_confusion_matrix(all_labels, all_preds, len(CLASSES))
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    print_confusion_matrix(confusion, CLASSES)
    
    metrics = compute_per_class_metrics(confusion, CLASSES)
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 70)
    
    for m in metrics:
        print(f"{m['class']:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1']:<12.4f} {m['support']:<10}")
    
    avg_precision = np.mean([m['precision'] for m in metrics])
    avg_recall = np.mean([m['recall'] for m in metrics])
    avg_f1 = np.mean([m['f1'] for m in metrics])
    
    print("-" * 70)
    print(f"{'Macro Avg':<20} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    print("=" * 60)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(",".join([""] + CLASSES) + "\n")
            for i, name in enumerate(CLASSES):
                row = [name] + [str(confusion[i, j]) for j in range(len(CLASSES))]
                f.write(",".join(row) + "\n")
            f.write("\nPer-Class Metrics:\n")
            f.write("Class,Precision,Recall,F1,Support\n")
            for m in metrics:
                f.write(f"{m['class']},{m['precision']:.4f},{m['recall']:.4f},"
                        f"{m['f1']:.4f},{m['support']}\n")
        
        print(f"\nResults saved to {output_path}")
    
    return accuracy, confusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], required=True,
                        help="Model type (must match checkpoint)")
    
    # Model architecture (must match training)
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (must match training)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of layers (must match training)")
    
    # Data arguments
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv",
                        help="Path to test CSV")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (optional)")
    
    args = parser.parse_args()
    evaluate(args)






