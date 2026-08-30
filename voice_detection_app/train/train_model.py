import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from voice_detection_app.config import settings
from voice_detection_app.models.detector import VoiceAuthenticityNet
from voice_detection_app.train.dataset_loader import CombinedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_training_data(
    num_samples: int = 2000,
    feature_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for bootstrapping or fallback."""
    np.random.seed(42)

    genuine = np.random.normal(0.5, 0.15, (num_samples // 2, feature_size)).clip(0, 1)
    synthetic = np.random.normal(0.5, 0.2, (num_samples // 2, feature_size)).clip(0, 1)
    synthetic += np.random.normal(0, 0.05, synthetic.shape)

    X = np.vstack([genuine, synthetic]).astype(np.float32)
    y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2), dtype=np.float32)

    indices = np.random.permutation(num_samples)
    return X[indices], y[indices]


def load_real_data(data_dir: str = "data", max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Attempt to load real datasets. Returns empty arrays if unavailable."""
    dataset = CombinedDataset(data_dir)
    X, y = dataset.prepare(
        max_asvspoof=max_samples,
        max_wavefake=max_samples,
    )
    if len(X) > 0:
        logger.info("Loaded %d real samples (%d genuine, %d synthetic)",
                     len(X), int(np.sum(y == 0)), int(np.sum(y == 1)))
    return X, y


def train(
    num_epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_dir: str = "data",
    use_real_data: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    if use_real_data:
        X, y = load_real_data(data_dir)
    else:
        X = np.array([])

    if len(X) == 0:
        logger.info("No real data available. Using synthetic training data.")
        X, y = generate_synthetic_training_data()

    split = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info("Train: %d samples | Val: %d samples", len(X_train), len(X_val))

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = VoiceAuthenticityNet(
        input_size=settings.model.input_features,
        hidden_sizes=settings.model.hidden_sizes,
        dropout=settings.model.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 20

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X).squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X).squeeze(-1)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                predicted = (torch.sigmoid(output) > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        train_acc = correct_train / total_train

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f",
                epoch + 1, num_epochs, avg_train_loss, train_acc, avg_val_loss, accuracy,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = settings.model.model_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Model saved to %s", settings.model.model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir,
        use_real_data=not args.synthetic_only,
    )
