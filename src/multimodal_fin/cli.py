# """
# Command-line interface for the multimodal_fin package.
# Defines the Typer app and entry points for both console_scripts and direct execution.
# """
# from pathlib import Path
# import typer

# from multimodal_fin.config import load_settings, load_data_settings
# from multimodal_fin.runners import ProcessRunner, EmbedRunner, DataAdquisitionRunner


# app = typer.Typer(help="Multimodal conference processing CLI.")


# @app.command()
# def process(
#     config_file: Path = typer.Option(..., help="Path to the YAML configuration file."),
#     config_name: str = typer.Option("default", help="Configuration section name in the YAML file."),
# ) -> None:
#     """
#     Execute the full pipeline: preprocessing, classification, and multimodal analysis.

#     Args:
#         config_file (Path): Path to the YAML file containing pipeline settings.
#         config_name (str): Key within the YAML under 'configs' indicating which settings to use.

#     Returns:
#         None
#     """
#     settings = load_settings(str(config_file), config_name)
#     runner = ProcessRunner(settings)
#     runner.run()



# @app.command()
# def embed(
#     config_file: Path = typer.Option(..., help="YAML de configuración"),
#     config_name: str = typer.Option("default", help="Sección dentro de configs"),
#     json_path: Path = typer.Option(None, help="Ruta única a transcript.json"),
#     json_csv: Path = typer.Option(None, help="CSV con lista de rutas a transcript.json"),
# ) -> None:
#     """
#     Genera embeddings desde JSON enriquecidos.

#     - Si indicas --json-path, procesa solo ese JSON.
#     - Si indicas --json-csv, procesa todos los JSON listados en el CSV.
#     """
#     settings = load_settings(str(config_file), config_name)
#     runner = EmbedRunner(settings)
#     runner.run(json_path=json_path, json_csv=json_csv)



# @app.command()
# def download(
#     config_file: Path = typer.Option(..., help="YAML de configuración"),
#     config_name: str = typer.Option("default", help="Nombre de sección en YAML"),
#     url: str = typer.Option("https://earningscall.biz/sp500", help="URL base para descargar datos de conferencias")
# ):
#     """
#     Descarga los datos de conferencias de earningscall.biz para empresas del SP500.
#     """
#     settings = load_data_settings(str(config_file), config_name, url)
#     runner = DataAdquisitionRunner(settings)
#     runner.run()


# def main() -> None:
#     """
#     Console-script entrypoint: invokes the Typer app.
#     """
#     app()


# if __name__ == "__main__":
#     main()
"""
Command-line interface for the multimodal_fin package.
Defines the Typer app and entry points for both console_scripts and direct execution.
"""

from pathlib import Path
import typer

from multimodal_fin.config import (
    load_settings,
    load_embed_settings,
    load_data_settings,
)
from multimodal_fin.runners import (
    ProcessRunner,
    EmbedRunner,
    DataAdquisitionRunner,
)


app = typer.Typer(help="Multimodal conference processing CLI.")


@app.command()
def process(
    config_file: Path = typer.Option(..., help="Path to the YAML configuration file."),
    config_name: str = typer.Option("default", help="Section name in 'conferences_processing'."),
) -> None:
    """
    Execute the full pipeline: preprocessing, classification, and multimodal analysis.
    """
    settings = load_settings(str(config_file), config_name)
    runner = ProcessRunner(settings)
    runner.run()


@app.command()
def embed(
    config_file: Path = typer.Option(..., help="Path to the YAML configuration file."),
    config_name: str = typer.Option("default", help="Section name in 'embeddings_pipeline'."),
    json_path: Path = typer.Option(None, help="Single path to transcript.json"),
    json_csv: Path = typer.Option(None, help="CSV with list of paths to transcript.json files."),
) -> None:
    """
    Generate embeddings from enriched JSONs.
    """
    settings = load_settings(str(config_file), config_name)
    embed_config = load_embed_settings(str(config_file), config_name)
    runner = EmbedRunner(settings, embed_config)
    runner.run(json_path=json_path, json_csv=json_csv)


@app.command()
def download(
    config_file: Path = typer.Option(..., help="Path to the YAML configuration file."),
    config_name: str = typer.Option("default", help="Section name in 'conferences_data_adquisition'."),
    url: str = typer.Option("https://earningscall.biz/sp-500-holdings", help="Optional override for S&P500 data URL."),
) -> None:
    """
    Download conference data from earningscall.biz for S&P500 companies.
    """
    settings = load_data_settings(str(config_file), config_name, override_url=url)
    runner = DataAdquisitionRunner(settings)
    runner.run()


def main() -> None:
    """
    Entry point when run as a console script.
    """
    app()


if __name__ == "__main__":
    main()