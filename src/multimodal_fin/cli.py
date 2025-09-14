"""
Command-line interface for the multimodal_fin package.

This script defines the main CLI entry points using Typer,
allowing users to:
- Process conference data
- Generate embeddings
- Download transcripts and audio

Each command loads its corresponding configuration section from a YAML file.
"""

from pathlib import Path
import typer

from multimodal_fin.config import load_full_config
from multimodal_fin.utils.cli import validate_embed_inputs
from multimodal_fin.runners import get_runner

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = typer.Typer(help="Multimodal conference processing CLI.")


@app.command()
def process(config_file: Path, config_name: str = "default") -> None:
    """Run the full pipeline: QA/monologue classification and enrichment.

    Args:
        config_file (Path): Path to the YAML configuration file.
        config_name (str, optional): Name of the config block under 'conferences_processing'. Defaults to "default".
    """
    config = load_full_config(str(config_file), config_name)
    runner = get_runner("process", config)
    runner.run()


@app.command()
def embed(
    config_file: Path,
    config_name: str = "default",
    json_path: Path = None,
    json_csv: Path = None
) -> None:
    """Generate hierarchical multimodal embeddings from enriched JSON files.

    Args:
        config_file (Path): Path to the YAML configuration file.
        config_name (str, optional): Name of the config block under 'embeddings_pipeline'. Defaults to "default".
        json_path (Path, optional): Path to a single `transcript.json` file.
        json_csv (Path, optional): Path to a CSV containing paths to multiple `transcript.json` files.
    """
    config = load_full_config(str(config_file), config_name)
    paths = validate_embed_inputs(json_path, json_csv)
    runner = get_runner("embed", config)
    runner.run(paths=paths)


@app.command()
def download(
    config_file: Path,
    config_name: str = "default",
    url: str = None
) -> None:
    """Download transcripts and audio from EarningsCall.biz for S&P500 companies.

    Args:
        config_file (Path): Path to the YAML configuration file.
        config_name (str, optional): Name of the config block under 'conferences_data_adquisition'. Defaults to "default".
        url (str, optional): Optional override of the default S&P500 earnings call URL.
    """
    config = load_full_config(str(config_file), config_name, override_url=url)
    runner = get_runner("download", config)
    runner.run()


def main() -> None:
    """Main entry point for the CLI when invoked directly."""
    app()


if __name__ == "__main__":
    main()