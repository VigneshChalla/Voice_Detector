"""
Download LibriSpeech dev-clean (diverse speakers) and combine with WaveFake synthetic.
This fixes the core issue: model trained only on LJ Speech (1 speaker) can't generalize.
"""
import logging
import os
import sys
import urllib.request
import zipfile
import soundfile as sf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LIBRISPEECH_ZIP = PROJECT_ROOT / "data" / "librispeech-dev-clean.tar.gz"
LIBRISPEECH_DIR = PROJECT_ROOT / "data" / "librispeech-dev-clean"
LIBRISPEECH_WAV = PROJECT_ROOT / "data" / "librispeech-genuine"


def download_librispeech():
    """Download LibriSpeech dev-clean if not already present."""
    if LIBRISPEECH_DIR.exists():
        logger.info("LibriSpeech already downloaded at %s", LIBRISPEECH_DIR)
        return True

    LIBRISPEECH_WAV.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading LibriSpeech dev-clean (~346 MB)...")
    logger.info("URL: %s", LIBRISPEECH_URL)

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            if pct % 10 == 0:
                logger.info("  Download progress: %d%%", pct)

    try:
        urllib.request.urlretrieve(LIBRISPEECH_URL, str(LIBRISPEECH_ZIP), progress)
    except Exception as e:
        logger.error("Download failed: %s", e)
        logger.info("Please download manually from: https://www.openslr.org/12")
        logger.info("Extract to: %s", LIBRISPEECH_DIR)
        return False

    logger.info("Extracting...")
    try:
        import tarfile
        with tarfile.open(str(LIBRISPEECH_ZIP), "r:gz") as tar:
            tar.extractall(str(PROJECT_ROOT / "data"))
        logger.info("Extracted to %s", LIBRISPEECH_DIR)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return False

    return True


def convert_to_wav():
    """Convert all FLAC files to WAV in librispeech-genuine/."""
    if not LIBRISPEECH_DIR.exists():
        logger.error("LibriSpeech directory not found: %s", LIBRISPEECH_DIR)
        return 0

    LIBRISPEECH_WAV.mkdir(parents=True, exist_ok=True)

    flac_files = list(LIBRISPEECH_DIR.rglob("*.flac"))
    logger.info("Found %d FLAC files to convert", len(flac_files))

    count = 0
    for i, flac_path in enumerate(flac_files):
        wav_name = f"{flac_path.parent.parent.name}_{flac_path.parent.name}_{flac_path.stem}.wav"
        wav_path = LIBRISPEECH_WAV / wav_name

        if wav_path.exists():
            count += 1
            continue

        try:
            data, sr = sf.read(str(flac_path))
            sf.write(str(wav_path), data, sr)
            count += 1
            if (i + 1) % 100 == 0:
                logger.info("  Converted %d/%d", i + 1, len(flac_files))
        except Exception as e:
            logger.warning("Failed to convert %s: %s", flac_path, e)

    logger.info("Converted %d WAV files to %s", count, LIBRISPEECH_WAV)
    return count


def main():
    logger.info("=" * 60)
    logger.info("Step 1: Download LibriSpeech dev-clean")
    logger.info("=" * 60)

    success = download_librispeech()
    if not success:
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Step 2: Convert FLAC to WAV")
    logger.info("=" * 60)

    wav_count = convert_to_wav()
    if wav_count == 0:
        logger.error("No WAV files created!")
        sys.exit(1)

    # Count files
    genuine_wavefake = list((PROJECT_ROOT / "data" / "wavefake" / "genuine").glob("*.wav"))
    genuine_libri = list(LIBRISPEECH_WAV.glob("*.wav"))
    synthetic = list((PROJECT_ROOT / "data" / "wavefake" / "synthetic").glob("*.wav"))

    logger.info("=" * 60)
    logger.info("DATA SUMMARY:")
    logger.info("  Genuine (WaveFake LJ Speech): %d", len(genuine_wavefake))
    logger.info("  Genuine (LibriSpeech diverse): %d", len(genuine_libri))
    logger.info("  Total genuine: %d", len(genuine_wavefake) + len(genuine_libri))
    logger.info("  Synthetic (WaveFake): %d", len(synthetic))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
