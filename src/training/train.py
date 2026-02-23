from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import conf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: torch.nn.Module,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    nr_of_epochs: int,
    batch_size: int = 64,
    lr: float = 0.002,
    weight_decay: float = 0.0001,
    seed: int = 42,
) -> None:
    loss_fn = torch.nn.CrossEntropyLoss()

    val_features = val_features.to(conf.device)
    val_labels = val_labels.to(conf.device).view(-1).long()

    train_labels = train_labels.view(-1).long()
    output_dim = model.L3.out_features  # matches your current model
    assert train_labels.min().item() >= 0
    assert train_labels.max().item() < output_dim

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=nr_of_epochs,
    )

    best_train = (0.0, -1)
    best_val = (0.0, -1)

    for epoch in range(nr_of_epochs):
        model.train()

        # determinism-ish
        set_seed(seed + epoch)

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for segments, labels in train_loader:
            segments = segments.to(conf.device)
            labels = labels.to(conf.device).view(-1).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(segments)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features)
            val_loss = loss_fn(val_logits, val_labels).item()
            val_acc = (val_logits.argmax(dim=1) == val_labels).float().mean().item()

        if train_acc > best_train[0]:
            best_train = (train_acc, epoch)
        if val_acc > best_val[0]:
            best_val = (val_acc, epoch)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
            f"Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}"
        )

    print(f"Done: best train acc={best_train[0]:.4f} @ epoch {best_train[1]}")
    print(f"Done: best   val acc={best_val[0]:.4f} @ epoch {best_val[1]}")
