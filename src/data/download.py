"""Download the CFPB Consumer Complaints dataset."""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import structlog

from src.utils.config import DATA_DIR, load_config

logger = structlog.get_logger(__name__)

RAW_DIR = DATA_DIR / "raw"


def download_dataset(config_path: str = "model_config.yaml") -> Path:
    """Download and extract the CFPB Consumer Complaints dataset.

    Source: https://www.consumerfinance.gov/data-research/consumer-complaints/
    """
    config = load_config(config_path)
    url = config["data"]["dataset_url"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / "complaints.csv"
    if output_path.exists():
        logger.info("Dataset already exists", path=str(output_path))
        return output_path

    zip_path = RAW_DIR / "complaints.csv.zip"
    logger.info("Downloading CFPB complaints dataset", url=url)

    try:
        urlretrieve(url, zip_path)
    except Exception as e:
        logger.error("Download failed", error=str(e))
        raise RuntimeError(
            f"Failed to download dataset from {url}. "
            "You can manually download from "
            "https://www.consumerfinance.gov/data-research/consumer-complaints/ "
            f"and place the CSV in {RAW_DIR}"
        ) from e

    logger.info("Extracting dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

    if not output_path.exists():
        # Handle case where zip contains differently named file
        csv_files = list(RAW_DIR.glob("*.csv"))
        if csv_files:
            csv_files[0].rename(output_path)

    logger.info("Dataset ready", path=str(output_path))
    return output_path


if __name__ == "__main__":
    download_dataset()
