import numpy as np
import torch
import torch.nn as nn

from voice_detection_app.config import settings


class VoiceAuthenticityNet(nn.Module):
    """Neural network for classifying voice as genuine or synthetic/cloned.
    
    V2 architecture: 128-dim input, GELU activation, deeper layers.
    """

    def __init__(self, input_size: int = 128, hidden_sizes: list[int] | None = None, dropout: float = 0.3):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 256, 256, 128, 64]

        layers = []
        prev_size = input_size
        for i, h_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.GELU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VoiceDetector:
    """High-level detector that wraps the neural network and preprocessing."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VoiceAuthenticityNet(
            input_size=settings.model.input_features,
            hidden_sizes=settings.model.hidden_sizes,
            dropout=settings.model.dropout,
        ).to(self.device)
        self.model.eval()
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, feature_vector: np.ndarray) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()

        return {
            "synthetic_probability": prob,
            "genuine_probability": 1.0 - prob,
            "is_synthetic": prob > 0.5,
        }

    def predict_batch(self, feature_vectors: np.ndarray) -> list[dict[str, float]]:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feature_vectors, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        results = []
        for prob in probs:
            results.append({
                "synthetic_probability": float(prob),
                "genuine_probability": float(1.0 - prob),
                "is_synthetic": prob > 0.5,
            })
        return results

    def load_model(self, path: str | None = None):
        path = path or settings.model.model_path
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()
            self._trained = True
        except FileNotFoundError:
            self._trained = False

    def save_model(self, path: str | None = None):
        path = path or settings.model.model_path
        torch.save(self.model.state_dict(), path)
