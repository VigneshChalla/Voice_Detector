"""Downloads ASVspoof 2021 and WaveFake datasets for training."""
import argparse
import hashlib
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ASVSPOOF_URLS = {
    "train_flac": "https://zenodo.org/record/4837280/files/ASVspoof2021Train.zip",
    "dev_flac": "https://zenodo.org/record/4837280/files/ASVspoof2021Dev.zip",
    "eval_flac": "https://zenodo.org/record/4837280/files/ASVspoof2021Eval.zip",
    "keys_train": "https://www.asvspoof.org/downloads/ASVspoof2021TrainKaldi.tar.gz",
    "keys_dev": "https://www.asvspoof.org/downloads/ASVspoof2021DevKaldi.tar.gz",
    "metadata": "https://www.asvspoof.org/downloads/LA-keys.zip",
}

WAVEFAKE_URL = "https://huggingface.co/datasets/RUB-NBS/WaveFake/resolve/main/WaveFake.zip"


def download_file(url: str, dest: Path, description: str = ""):
    if dest.exists():
        logger.info("Already exists: %s, skipping", dest.name)
        return dest

    logger.info("Downloading %s from %s", description or dest.name, url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            check=True,
            capture_output=True,
        )
        logger.info("Downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    except FileNotFoundError:
        logger.info("curl not found, trying powershell...")
        subprocess.run(
            ["powershell", "-Command", f"Invoke-WebRequest -Uri '{url}' -OutFile '{dest}'"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Download failed for %s: %s", dest.name, e.stderr.decode() if e.stderr else e)
        raise

    return dest


def extract_zip(zip_path: Path, dest_dir: Path):
    if not zip_path.exists():
        return
    logger.info("Extracting %s", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted to %s", dest_dir)


def download_asvspoof(output_dir: Path):
    logger.info("=== Downloading ASVspoof 2021 ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, url in ASVSPOOF_URLS.items():
        filename = url.split("/")[-1]
        dest = output_dir / filename
        download_file(url, dest, f"ASVspoof {key}")

        if filename.endswith(".zip") or filename.endswith(".tar.gz"):
            extract_dir = output_dir / filename.replace(".zip", "").replace(".tar.gz", "")
            if not extract_dir.exists():
                extract_zip(dest, output_dir)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    logger.info("ASVspoof 2021 download complete. Data at: %s", output_dir)


def download_wavefake(output_dir: Path):
    logger.info("=== Downloading WaveFake ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = WAVEFAKE_URL.split("/")[-1]
    dest = output_dir / filename
    download_file(WAVEFAKE_URL, dest, "WaveFake")
    extract_zip(dest, output_dir)
    logger.info("WaveFake download complete. Data at: %s", output_dir)


def download_all(output_dir: Path | None = None):
    output_dir = output_dir or DATA_DIR
    download_asvspoof(output_dir / "asvspoof")
    download_wavefake(output_dir / "wavefake")
    logger.info("=== All datasets downloaded to %s ===", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download voice cloning detection datasets")
    parser.add_argument("--dataset", choices=["asvspoof", "wavefake", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else DATA_DIR
    if args.dataset == "asvspoof":
        download_asvspoof(out / "asvspoof")
    elif args.dataset == "wavefake":
        download_wavefake(out / "wavefake")
    else:
        download_all(out)
