import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np

from mlp import MLP
from cnn import CNN


CLASSES = ["stop_sign", "yield_sign", "speed_limit_sign", "apple", "banana", "orange"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class ImageDataset(Dataset):
    # loads images listed in a CSV
    
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filepath": row["filepath"],
                    "label": CLASS_TO_IDX[row["label"]],
                    "label_name": row["label"]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["filepath"]).convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
        return img_tensor, sample["label"]
    
    def get_stratified_subset(self, n_per_class):
        # returns indices for n images per class
        class_indices = {c: [] for c in CLASSES}
        
        for idx, sample in enumerate(self.samples):
            class_indices[sample["label_name"]].append(idx)
        
        selected_indices = []
        for class_name in CLASSES:
            indices = class_indices[class_name]
            np.random.shuffle(indices)
            selected_indices.extend(indices[:n_per_class])
        
        return selected_indices


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / total, correct / total, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_acc, filepath, history=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }
    if history:
        checkpoint["history"] = history
    torch.save(checkpoint, filepath)


def log_experiment(log_path, experiment_data):
    file_exists = Path(log_path).exists()
    
    with open(log_path, "a", newline="") as f:
        fieldnames = ["experiment_id", "model", "epochs", "learning_rate", 
                      "batch_size", "hidden_dim", "num_layers", "dropout",
                      "best_epoch", "train_acc", "val_acc", "images_per_class",
                      "timestamp", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(experiment_data)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading datasets...")
    train_dataset = ImageDataset(args.train_csv)
    val_dataset = ImageDataset(args.val_csv)
    
    if args.images_per_class:
        np.random.seed(args.seed)
        train_indices = train_dataset.get_stratified_subset(args.images_per_class)
        train_dataset = Subset(train_dataset, train_indices)
        print(f"Using {args.images_per_class} images per class ({len(train_dataset)} total)")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == "mlp":
        model = MLP(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_layers,
            num_classes=len(CLASSES),
            dropout=args.dropout
        )
    else:
        model = CNN(
            base_filters=args.hidden_dim,
            fc_dim=512,
            dropout=args.dropout,
            num_classes=len(CLASSES)
        )
    
    model = model.to(device)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_train_acc = 0
    best_preds = None
    best_labels = None
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                             criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            patience_counter = 0
            best_preds = val_preds
            best_labels = val_labels
            
            checkpoint_name = f"best_{args.model}"
            if args.images_per_class:
                checkpoint_name += f"_{args.images_per_class}img"
            checkpoint_path = Path(args.checkpoint_dir) / f"{checkpoint_name}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path, history)
            print(f"  -> New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best train accuracy: {best_train_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 60)
    
    history_name = f"history_{args.model}"
    if args.images_per_class:
        history_name += f"_{args.images_per_class}img"
    history_path = Path(args.checkpoint_dir) / f"{history_name}.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Training history saved to {history_path}")
    
    if best_preds and best_labels:
        confusion_data = {
            "predictions": [int(x) for x in best_preds],
            "labels": [int(x) for x in best_labels],
            "classes": CLASSES
        }
        confusion_name = f"confusion_{args.model}"
        if args.images_per_class:
            confusion_name += f"_{args.images_per_class}img"
        confusion_path = Path(args.checkpoint_dir) / f"{confusion_name}.json"
        with open(confusion_path, "w") as f:
            json.dump(confusion_data, f)
    
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_data = {
        "experiment_id": experiment_id,
        "model": args.model,
        "epochs": best_epoch,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout if args.model == "cnn" else "N/A",
        "best_epoch": best_epoch,
        "train_acc": f"{best_train_acc:.4f}",
        "val_acc": f"{best_val_acc:.4f}",
        "images_per_class": args.images_per_class if args.images_per_class else "all",
        "timestamp": datetime.now().isoformat(),
        "notes": args.notes
    }
    
    log_experiment(args.log_path, experiment_data)
    print(f"\nExperiment logged to {args.log_path}")
    
    return best_val_acc, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP or CNN classifier")
    
    # Data arguments
    parser.add_argument("--train-csv", type=str, default="data/splits/train.csv",
                        help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default="data/splits/val.csv",
                        help="Path to validation CSV")
    parser.add_argument("--images-per-class", type=int, default=None,
                        help="Number of images per class (for subset experiments)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model arguments
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp",
                        help="Model type (mlp or cnn)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (MLP) or base filters (CNN)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of hidden/conv layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability (CNN only)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-path", type=str, 
                        default="experiments/experiment_log.csv",
                        help="Path to experiment log CSV")
    parser.add_argument("--notes", type=str, default="",
                        help="Notes for this experiment")
    
    args = parser.parse_args()
    train(args)
