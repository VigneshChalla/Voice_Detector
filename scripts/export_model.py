"""Export trained model to ONNX/TorchScript for edge deployment."""
import argparse
import logging

from voice_detection_app.config import settings
from voice_detection_app.edge.export_and_infer import EdgeModelExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export voice detection model for edge inference")
    parser.add_argument("--model-path", type=str, default=None, help="Path to PyTorch weights")
    parser.add_argument("--output-dir", type=str, default="voice_detection_app", help="Output directory")
    parser.add_argument("--format", choices=["onnx", "torchscript", "quantized", "all"], default="all")
    args = parser.parse_args()

    exporter = EdgeModelExporter()
    exporter.load_pytorch_weights(args.model_path)

    if args.format in ("onnx", "all"):
        path = exporter.export_onnx(f"{args.output_dir}/trained_model.onnx")
        logger.info("ONNX exported: %s", path)

    if args.format in ("torchscript", "all"):
        path = exporter.export_torchscript(f"{args/output_dir}/trained_model.pt")
        logger.info("TorchScript exported: %s", path)

    if args.format in ("quantized", "all"):
        path = exporter.quantize_model(f"{args.output_dir}/trained_model_quantized.pt")
        logger.info("Quantized exported: %s", path)

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
