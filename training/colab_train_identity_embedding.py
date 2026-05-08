from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a crop-level pill identity embedding model.")
    parser.add_argument("--manifest", required=True, help="CSV with label and image_path columns.")
    parser.add_argument("--image-root", help="Root used to resolve relative image_path values.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--model", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and torchvision first.") from exc

    rows = read_manifest(
        Path(args.manifest),
        image_root=None if args.image_root is None else Path(args.image_root),
        label_column=args.label_column,
    )
    train_rows, val_rows = split_rows_by_label(rows, args.val_fraction, args.seed)
    labels = sorted({row["label"] for row in train_rows})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size + 32, args.img_size + 32)),
            transforms.RandomResizedCrop(args.img_size, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    class CropDataset(Dataset):
        def __init__(self, dataset_rows: list[dict[str, str]], transform) -> None:
            self.rows = [row for row in dataset_rows if row["label"] in label_to_index]
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            image = Image.open(row["image_path"]).convert("RGB")
            return self.transform(image), label_to_index[row["label"]]

    train_loader = DataLoader(
        CropDataset(train_rows, train_tfms),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        CropDataset(val_rows, val_tfms),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class IdentityEmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone, feature_dim = build_backbone(args.model, models, nn)
            self.embedding = nn.Sequential(
                nn.Linear(feature_dim, args.embedding_dim),
                nn.BatchNorm1d(args.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.embedding_dim, args.embedding_dim),
            )
            self.classifier = nn.Linear(args.embedding_dim, len(label_to_index))

        def forward(self, images):
            features = self.backbone(images)
            embeddings = F.normalize(self.embedding(features), dim=1)
            logits = self.classifier(embeddings) * 16.0
            return embeddings, logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdentityEmbeddingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_acc = 0.0
    history: list[dict[str, float]] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, torch, nn)
        val_acc, val_loss = evaluate(model, val_loader, device, torch, nn)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1": val_acc}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": args.model,
                    "embedding_dim": args.embedding_dim,
                    "img_size": args.img_size,
                    "label_to_index": label_to_index,
                    "index_to_label": {index: label for label, index in label_to_index.items()},
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                output_dir / "identity_embedding.pt",
            )

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "labels": len(label_to_index),
        "best_val_top1": best_acc,
        "model_path": str(output_dir / "identity_embedding.pt"),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def read_manifest(manifest: Path, image_root: Path | None, label_column: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if "image_path" not in (reader.fieldnames or []) or label_column not in (reader.fieldnames or []):
            raise ValueError(f"Manifest must contain image_path and {label_column} columns.")
        for row in reader:
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (image_root or manifest.parent) / image_path
            if image_path.exists():
                rows.append({"image_path": str(image_path), "label": row[label_column]})
    if not rows:
        raise ValueError("No valid identity training rows were found.")
    return rows


def split_rows_by_label(
    rows: list[dict[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        if len(label_rows) < 3:
            train_rows.extend(label_rows)
            continue
        val_count = max(1, round(len(label_rows) * val_fraction))
        val_rows.extend(label_rows[:val_count])
        train_rows.extend(label_rows[val_count:])
    return train_rows, val_rows


def build_backbone(name: str, models_module: Any, nn_module: Any):
    if name == "mobilenet_v3_small":
        weights = models_module.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models_module.mobilenet_v3_small(weights=weights)
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn_module.Identity()
        return backbone, feature_dim
    if name == "resnet18":
        weights = models_module.ResNet18_Weights.DEFAULT
        backbone = models_module.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn_module.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported backbone: {name}")


def train_one_epoch(model, loader, optimizer, scaler, device, torch_module: Any, nn_module: Any) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    criterion = nn_module.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch_module.cuda.amp.autocast(enabled=device.type == "cuda"):
            _, logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


def evaluate(model, loader, device, torch_module: Any, nn_module: Any) -> tuple[float, float]:
    model.eval()
    criterion = nn_module.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch_module.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, logits = model(images)
            loss = criterion(logits, labels)
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * images.size(0)
            total += images.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)


if __name__ == "__main__":
    raise SystemExit(main())
