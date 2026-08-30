"""ONNX export and lightweight edge inference for on-device deployment."""
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from voice_detection_app.config import settings
from voice_detection_app.models.detector import VoiceAuthenticityNet

logger = logging.getLogger(__name__)


class EdgeModelExporter:
    """Exports the trained model to ONNX format for edge/device inference."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = VoiceAuthenticityNet(
            input_size=settings.model.input_features,
            hidden_sizes=settings.model.hidden_sizes,
            dropout=0.0,
        ).to(self.device)

    def load_pytorch_weights(self, path: str | None = None):
        path = path or settings.model.model_path
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Loaded PyTorch weights from %s", path)

    def export_onnx(self, output_path: str = "voice_detection_app/trained_model.onnx", opset_version: int = 14):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(1, settings.model.input_features, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["features"],
            output_names=["synthetic_prob"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "synthetic_prob": {0: "batch_size"},
            },
        )
        logger.info("ONNX model exported to %s", output_path)
        return str(output_path)

    def export_torchscript(self, output_path: str = "voice_detection_app/trained_model.pt"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(1, settings.model.input_features, device=self.device)
        traced = torch.jit.trace(self.model, dummy_input)
        traced.save(str(output_path))
        logger.info("TorchScript model exported to %s", output_path)
        return str(output_path)

    def quantize_model(self, output_path: str = "voice_detection_app/trained_model_quantized.pt"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        quantized = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            torch.qint8,
        )
        dummy_input = torch.randn(1, settings.model.input_features)
        traced = torch.jit.trace(quantized, dummy_input)
        traced.save(str(output_path))
        logger.info("Quantized model exported to %s", output_path)
        return str(output_path)


class EdgeInferenceEngine:
    """Lightweight inference engine for edge/device deployment using ONNX Runtime."""

    def __init__(self, onnx_path: str | None = None):
        self.onnx_path = onnx_path or "voice_detection_app/trained_model.onnx"
        self.session = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                self.onnx_path,
                providers=["CPUExecutionProvider"],
            )
            self._available = True
            logger.info("ONNX Runtime loaded model from %s", self.onnx_path)
        except ImportError:
            logger.warning("onnxruntime not installed. Edge inference unavailable.")
        except Exception as e:
            logger.warning("Failed to load ONNX model: %s", e)

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(self, feature_vector: np.ndarray) -> dict[str, float]:
        if not self._available:
            raise RuntimeError("ONNX model not loaded. Run export first or install onnxruntime.")

        input_data = feature_vector.reshape(1, -1).astype(np.float32)
        outputs = self.session.run(None, {"features": input_data})
        prob = float(outputs[0].flatten()[0])

        return {
            "synthetic_probability": prob,
            "genuine_probability": 1.0 - prob,
            "is_synthetic": prob > 0.5,
        }

    def predict_batch(self, feature_vectors: np.ndarray) -> list[dict[str, float]]:
        if not self._available:
            raise RuntimeError("ONNX model not loaded.")

        input_data = feature_vectors.astype(np.float32)
        outputs = self.session.run(None, {"features": input_data})
        probs = outputs[0].flatten()

        return [
            {
                "synthetic_probability": float(p),
                "genuine_probability": float(1.0 - p),
                "is_synthetic": p > 0.5,
            }
            for p in probs
        ]
