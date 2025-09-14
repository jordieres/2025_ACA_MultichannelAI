"""
CLI-related utility functions for validating user input in multimodal_fin.

This module provides validation logic for CLI arguments, specifically
used when embedding enriched JSON files.
"""

from pathlib import Path
from typing import List, Optional

import typer
import pandas as pd


def validate_embed_inputs(json_path: Optional[Path] = None, json_csv: Optional[Path] = None) -> List[str]:
    """Validate and resolve the JSON file paths to be embedded.

    This function ensures that exactly one of the two options is provided:
    either a single JSON path or a CSV file containing multiple paths.

    Args:
        json_path (Optional[Path]): Path to a single enriched transcript JSON file.
        json_csv (Optional[Path]): Path to a CSV containing a 'Paths' column.

    Returns:
        List[str]: A list of file paths to be processed.

    Raises:
        typer.Exit: If both or neither arguments are provided, or if CSV reading fails.
    """
    if json_path and json_csv:
        typer.echo("⚠️ Please provide either --json-path or --json-csv, not both.", err=True)
        raise typer.Exit(1)

    if not (json_path or json_csv):
        typer.echo("⚠️ You must provide either --json-path or --json-csv.", err=True)
        raise typer.Exit(1)

    if json_csv:
        try:
            df = pd.read_csv(json_csv)
            return df['Paths'].dropna().astype(str).tolist()
        except Exception as e:
            typer.echo(f"❌ Failed to read CSV file: {e}", err=True)
            raise typer.Exit(1)

    return [str(json_path)]
