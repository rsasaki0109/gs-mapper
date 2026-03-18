"""Dataset download utilities.

This module provides functions to download datasets from various sources
(HuggingFace Hub, Google Drive, direct URLs) and organize them into the
expected directory structure under ``data/``.

Supported download backends:
- huggingface_hub: For datasets hosted on HuggingFace
- gdown: For Google Drive links
- urllib/requests: For direct HTTP(S) URLs
- git clone: For Git repositories
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from nerf_gs_playground.common.config import get_dataset_config, get_project_root

logger = logging.getLogger(__name__)


def download_dataset(
    name: str,
    output_dir: Path | str | None = None,
    max_samples: int | None = None,
) -> Path:
    """Download a dataset by name to the destination directory.

    Args:
        name: Dataset identifier (e.g. "ggrt", "covla", "mcd").
        output_dir: Root directory where the dataset will be stored.
            Defaults to ``<project_root>/data/<name>``.
        max_samples: Maximum number of samples to download (for partial downloads).

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        ValueError: If the dataset name is not recognized.
        RuntimeError: If the download fails.
    """
    config = get_dataset_config(name)

    if output_dir is None:
        output_dir = get_project_root() / "data" / name
    else:
        output_dir = Path(output_dir) / name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if _is_downloaded(output_dir):
        logger.info("Dataset '%s' already exists at %s, skipping download.", name, output_dir)
        print(f"Dataset '{name}' already exists at {output_dir}, skipping download.")
        return output_dir

    print(f"Downloading dataset '{name}' to {output_dir}...")

    # Determine download method based on config
    if "huggingface" in config:
        _download_huggingface(config["huggingface"], output_dir, max_samples=max_samples)
    elif "download_url" in config:
        _download_url(config["download_url"], output_dir)
    elif "repository" in config:
        _download_git_clone(config["repository"], output_dir)
    elif "sample_url" in config:
        _download_url(config["sample_url"], output_dir)
    else:
        # For datasets without direct download URLs, provide instructions
        _create_download_instructions(name, config, output_dir)

    print(f"Dataset '{name}' downloaded to {output_dir}")
    return output_dir


def download_sample_images(output_dir: Path | str, num_images: int = 10) -> Path:
    """Download a small set of sample images for quick testing.

    Downloads sample images from a public domain source for demo purposes.

    Args:
        output_dir: Directory where images will be saved.
        num_images: Number of sample images to download.

    Returns:
        Path to the directory containing downloaded images.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if any(images_dir.iterdir()):
        print(f"Sample images already exist at {images_dir}, skipping download.")
        return images_dir

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    # Use picsum.photos for random public domain images
    base_url = "https://picsum.photos/800/600"

    items = range(num_images)
    if tqdm is not None:
        items = tqdm(items, desc="Downloading sample images")

    downloaded = 0
    for i in items:
        out_path = images_dir / f"frame_{i:04d}.jpg"
        if out_path.exists():
            downloaded += 1
            continue
        try:
            urllib.request.urlretrieve(f"{base_url}?random={i}", str(out_path))
            downloaded += 1
        except Exception as e:
            logger.warning("Failed to download sample image %d: %s", i, e)

    print(f"Downloaded {downloaded}/{num_images} sample images to {images_dir}")
    return images_dir


def _is_downloaded(output_dir: Path) -> bool:
    """Check if a dataset directory has content (not empty beyond .gitkeep)."""
    if not output_dir.exists():
        return False
    contents = [p for p in output_dir.iterdir() if p.name != ".gitkeep"]
    return len(contents) > 0


def _download_huggingface(
    repo_id: str,
    output_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Download a dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g. "tier4/CoVLA_Dataset").
        output_dir: Directory where the dataset will be saved.
        max_samples: Maximum number of files to download.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to download this dataset. "
            "Install it with: pip install huggingface-hub"
        )

    try:
        from tqdm import tqdm
    except ImportError:
        pass

    print(f"Downloading from HuggingFace: {repo_id}")

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(output_dir),
    }

    if max_samples is not None:
        # For partial downloads, use allow_patterns to limit files
        kwargs["allow_patterns"] = "*.jpg"

    try:
        snapshot_download(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to download from HuggingFace '{repo_id}': {e}")


def _download_url(url: str, output_dir: Path) -> None:
    """Download a file from a direct URL.

    Supports .tar.gz, .zip archives which will be extracted automatically.

    Args:
        url: Direct download URL.
        output_dir: Directory where the file will be saved/extracted.
    """
    filename = url.split("/")[-1].split("?")[0]
    if not filename:
        filename = "download"
    dest_path = output_dir / filename

    try:
        from tqdm import tqdm

        # Download with progress bar
        print(f"Downloading {url}")
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get("Content-Length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    except ImportError:
        print(f"Downloading {url} (install tqdm for progress bar)")
        urllib.request.urlretrieve(url, str(dest_path))

    # Extract if archive
    if filename.endswith((".tar.gz", ".tgz")):
        import tarfile

        print(f"Extracting {filename}...")
        with tarfile.open(dest_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        dest_path.unlink()
    elif filename.endswith(".zip"):
        import zipfile

        print(f"Extracting {filename}...")
        with zipfile.ZipFile(dest_path, "r") as z:
            z.extractall(output_dir)
        dest_path.unlink()


def _download_gdown(file_id: str, output_dir: Path) -> None:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID.
        output_dir: Directory where the file will be saved.
    """
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to download from Google Drive. "
            "Install it with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive: {file_id}")
    gdown.download(url, str(output_dir / "download"), quiet=False)


def _download_git_clone(repo_url: str, output_dir: Path) -> None:
    """Clone a Git repository.

    Args:
        repo_url: Git repository URL.
        output_dir: Directory where the repository will be cloned.
    """
    # Clone to a temporary subdirectory
    clone_dir = output_dir / "repo"
    if clone_dir.exists():
        print(f"Repository already cloned at {clone_dir}")
        return

    print(f"Cloning repository: {repo_url}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("git is not installed. Please install git to clone repositories.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e.stderr}")


def _create_download_instructions(name: str, config: dict[str, Any], output_dir: Path) -> None:
    """Create a README with download instructions for datasets without direct URLs.

    Args:
        name: Dataset name.
        config: Dataset configuration dictionary.
        output_dir: Output directory.
    """
    instructions = f"# Download Instructions for {config.get('name', name)}\n\n"
    instructions += f"{config.get('description', 'No description available.')}\n\n"

    if "paper" in config:
        instructions += f"Paper: {config['paper']}\n"
    if "repository" in config:
        instructions += f"Repository: {config['repository']}\n"
    if "url" in config:
        instructions += f"Website: {config['url']}\n"

    if "datasets" in config:
        instructions += "\n## Available sub-datasets:\n"
        for sub_name, sub_config in config["datasets"].items():
            instructions += f"\n### {sub_name}\n"
            instructions += f"- Description: {sub_config.get('description', 'N/A')}\n"
            instructions += f"- URL: {sub_config.get('url', 'N/A')}\n"

    instructions += (
        f"\nPlease download the data manually and place it in:\n  {output_dir}\n"
    )

    readme_path = output_dir / "DOWNLOAD_INSTRUCTIONS.md"
    readme_path.write_text(instructions)
    print(f"Download instructions written to {readme_path}")
    print("This dataset requires manual download. See the instructions file for details.")
