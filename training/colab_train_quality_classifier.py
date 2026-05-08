from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OK/NG crop quality classifier.")
    parser.add_argument("--data-root", required=True, help="Folder with train/ and val/ ImageFolder layout.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        from torchvision import datasets, models
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and torchvision first.") from exc

    data_root = Path(args.data_root).resolve()
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Quality data must use data-root/train and data-root/val folders.")

    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size + 32, args.img_size + 32)),
            transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
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

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_classifier(args.model, len(train_ds.classes), models, nn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, torch)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device, torch)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1": val_acc}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": args.model,
                    "img_size": args.img_size,
                    "classes": train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                output_dir / "quality_classifier.pt",
            )

    summary = {
        "data_root": str(data_root),
        "classes": train_ds.classes,
        "best_val_top1": best_acc,
        "model_path": str(output_dir / "quality_classifier.pt"),
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_classifier(name: str, class_count: int, models_module: Any, nn_module: Any):
    if name == "mobilenet_v3_small":
        weights = models_module.MobileNet_V3_Small_Weights.DEFAULT
        model = models_module.mobilenet_v3_small(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn_module.Linear(feature_dim, class_count)
        return model
    if name == "resnet18":
        weights = models_module.ResNet18_Weights.DEFAULT
        model = models_module.resnet18(weights=weights)
        model.fc = nn_module.Linear(model.fc.in_features, class_count)
        return model
    raise ValueError(f"Unsupported classifier backbone: {name}")


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, torch_module: Any) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch_module.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


def evaluate(model, loader, criterion, device, torch_module: Any) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch_module.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * images.size(0)
            total += images.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)


if __name__ == "__main__":
    raise SystemExit(main())
