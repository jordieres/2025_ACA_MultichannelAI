"""
File and path utilities for the multimodal_fin package.

Includes reusable helpers for reading CSV and JSON files,
and locating specific files within conference directories.
"""

from pathlib import Path
from typing import List, Any
import pandas as pd
import json


def read_paths_csv(csv_path: str) -> List[str]:
    """Read a CSV file with a 'path' column and return a list of valid paths.

    Args:
        csv_path (str): Path to the CSV file containing a 'path' column.

    Returns:
        List[str]: List of paths to directories or files.

    Raises:
        ValueError: If the 'path' column is missing from the CSV.
    """
    df = pd.read_csv(csv_path)
    if 'path' not in df.columns:
        raise ValueError("Input CSV must contain a 'path' column.")
    return df['path'].dropna().tolist()


def make_processed_path(original: Path) -> Path:
    """Generate the processed output path from an original conference path.

    If the original path contains a folder named 'companies', it will be replaced
    with 'processed_companies'. Otherwise, the method appends '_processed' to the
    directory name under the same parent.

    Args:
        original (Path): Original input directory path.

    Returns:
        Path: Transformed path pointing to processed data.
    """
    parts = list(original.parts)
    try:
        idx = parts.index('companies')
        parts[idx] = 'processed_companies'
        return Path(*parts)
    except ValueError:
        return original.parent / f"{original.name}_processed"


def read_json_file(json_path: Path) -> Any:
    """Read a JSON file and return its parsed content.

    Args:
        json_path (Path): Full path to a JSON file.

    Returns:
        Any: Parsed JSON content (usually a dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_level3_json(directory: Path) -> Path:
    """Locate a LEVEL_3.json file in a given directory.

    Args:
        directory (Path): Directory to search in.

    Returns:
        Path: Full path to the LEVEL_3.json file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    candidate = directory / 'LEVEL_3.json'
    if not candidate.exists():
        raise FileNotFoundError(f"LEVEL_3.json not found in {directory}")
    return candidate


def find_audio_file(directory: Path) -> Path:
    """Locate the first audio file in a directory (supports mp3, wav, flac).

    Args:
        directory (Path): Directory to search in.

    Returns:
        Path: Full path to the first found audio file.

    Raises:
        FileNotFoundError: If no supported audio file is found.
    """
    for ext in ('mp3', 'wav', 'flac'):
        for file in directory.glob(f'*.{ext}'):
            return file
    raise FileNotFoundError(f"No audio file found in {directory}")
