"""Dataset loading and preprocessing for ASVspoof 2021 and WaveFake."""
import logging
from pathlib import Path

import numpy as np

from voice_detection_app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class ASVspoofDataset:
    """Loads and preprocesses ASVspoof 2021 dataset."""

    LABELS_MAP = {"spoof": 1, "bonafide": 0}

    def __init__(self, data_dir: str = "data/asvspoof"):
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor()
        self.file_index: dict[str, list[Path]] = {}

    def build_index(self):
        for ext in ("*.flac", "*.wav"):
            for path in self.data_dir.rglob(ext):
                parent = path.parent.name.lower()
                if parent not in self.file_index:
                    self.file_index[parent] = []
                self.file_index[parent].append(path)

        logger.info("ASVspoof index built: %d total files", sum(len(v) for v in self.file_index.values()))
        return self

    def load_labels(self, label_file: str = "") -> dict[str, int]:
        labels = {}
        if not label_file:
            for lf in self.data_dir.rglob("*labels*"):
                label_file = str(lf)
                break

        if label_file and Path(label_file).exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = Path(parts[0]).stem
                        label = self.LABELS_MAP.get(parts[-1], -1)
                        if label >= 0:
                            labels[filename] = label
            logger.info("Loaded %d labels from %s", len(labels), label_file)

        return labels

    def extract_features(
        self,
        max_files: int | None = None,
        labels: dict[str, int] | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        features = []
        labels_out = []
        count = 0

        for category, paths in self.file_index.items():
            for path in paths:
                if max_files and count >= max_files:
                    break

                try:
                    y, sr = self.audio_processor.load_audio(path)
                    if len(y) < sr:
                        continue
                    _, agg = self.audio_processor.process_audio(y)
                    fv = self.audio_processor.get_feature_vector(agg, target_length=64)
                    features.append(fv)

                    if labels:
                        label = labels.get(path.stem, -1)
                    else:
                        label = self.LABELS_MAP.get(category, 0)
                    labels_out.append(label)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to process %s: %s", path, e)

        logger.info("Extracted features from %d files", len(features))
        return features, labels_out


class WaveFakeDataset:
    """Loads and preprocesses WaveFake dataset."""

    def __init__(self, data_dir: str = "data/wavefake"):
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor()

    def find_audio_files(self) -> list[Path]:
        files = []
        for ext in ("*.wav", "*.mp3", "*.flac"):
            files.extend(self.data_dir.rglob(ext))
        logger.info("Found %d audio files in WaveFake", len(files))
        return files

    def extract_features(self, max_files: int | None = None) -> tuple[list[np.ndarray], list[int]]:
        files = self.find_audio_files()
        features = []
        labels = []
        count = 0

        for path in files:
            if max_files and count >= max_files:
                break

            try:
                y, sr = self.audio_processor.load_audio(path)
                if len(y) < sr:
                    continue
                _, agg = self.audio_processor.process_audio(y)
                fv = self.audio_processor.get_feature_vector(agg, target_length=64)
                features.append(fv)
                labels.append(1)
                count += 1
            except Exception as e:
                logger.warning("Failed to process %s: %s", path, e)

        logger.info("Extracted %d synthetic features from WaveFake", len(features))
        return features, labels


class CombinedDataset:
    """Merges multiple datasets for unified training."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.asvspoof = ASVspoofDataset(str(self.data_dir / "asvspoof"))
        self.wavefake = WaveFakeDataset(str(self.data_dir / "wavefake"))

    def prepare(
        self,
        max_asvspoof: int | None = 5000,
        max_wavefake: int | None = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []

        try:
            self.asvspoof.build_index()
            labels = self.asvspoof.load_labels()
            feats, labs = self.asvspoof.extract_features(max_asvspoof, labels)
            all_features.extend(feats)
            all_labels.extend(labs)
        except Exception as e:
            logger.warning("ASVspoof loading failed: %s. Using synthetic data.", e)

        try:
            feats, labs = self.wavefake.extract_features(max_wavefake)
            all_features.extend(feats)
            all_labels.extend(labs)
        except Exception as e:
            logger.warning("WaveFake loading failed: %s. Using synthetic data.", e)

        if not all_features:
            logger.warning("No real data loaded. Falling back to synthetic training data.")
            return np.array([]), np.array([])

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)

        perm = np.random.permutation(len(X))
        return X[perm], y[perm]
