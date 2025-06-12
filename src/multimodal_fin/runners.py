from pathlib import Path
import typer
import pandas as pd
import earningscall
from bs4 import BeautifulSoup
import requests

from multimodal_fin.config import (
    load_settings,
    load_embed_settings,
    load_data_settings,
    Settings,
    EmbeddingsPipelineSettings,
    DataAdquisitionSettings
)
from multimodal_fin.processors.conference import ConferenceProcessor
from multimodal_fin.embeddings.ConferencePipeline import ConferenceEmbeddingPipeline
from multimodal_fin.data_adquisition.Company import CompanyDataAcquisition


class DataAdquisitionRunner:
    def __init__(self, settings: DataAdquisitionSettings):
        self.settings = settings

    def run(self) -> None:
        """
        Ejecuta el pipeline de adquisición de datos desde earningscall.biz.
        """
        earningscall.api_key = self.settings.api_key

        url = self.settings.url
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        headers = [header.text.strip() for header in table.find_all('th')]
        rows = [
            [col.text.strip() for col in row.find_all('td')]
            for row in table.find_all('tr')[1:]
        ]
        df = pd.DataFrame(rows, columns=headers)
        SP500_data = df.groupby("Sector").head(8).reset_index(drop=True)

        for company_code in SP500_data['Symbol']:
            company = CompanyDataAcquisition(company_code)
            company.get_and_save_all_transcripts_and_audio(self.settings.base_path)


class ProcessRunner:
    def __init__(self, settings: Settings):
        self.processor = ConferenceProcessor(settings)

    def run(self) -> None:
        """
        Ejecuta el pipeline completo de texto y multimodal.
        """
        self.processor.run()


class EmbedRunner:
    def __init__(self, settings: Settings, emb_cfg: EmbeddingsPipelineSettings):
        self.pipeline = ConferenceEmbeddingPipeline(
            node_encoder_params=emb_cfg.node_encoder.model_dump(),
            conference_encoder_params=emb_cfg.conference_encoder.model_dump(),
            device=emb_cfg.device or settings.device,
        )
        self.pipeline.node_encoder.eval()
        self.pipeline.conference_encoder.eval()

    def run(self, json_path: Path = None, json_csv: Path = None) -> None:
        """
        Genera embeddings desde uno o varios JSON.
        """
        if json_path and json_csv:
            typer.echo("⚠️ Elige solo --json-path o --json-csv, no ambos", err=True)
            raise typer.Exit(1)
        if not (json_path or json_csv):
            typer.echo("⚠️ Debes indicar --json-path o --json-csv", err=True)
            raise typer.Exit(1)

        paths = [json_path] if json_path else list(pd.read_csv(json_csv)['Paths'])

        for p in paths:
            try:
                emb = self.pipeline.generate_embedding(str(p), return_attn=True)
                arr = emb.detach().cpu().numpy().flatten()
                typer.echo(' ')
                typer.echo('-' * 40)
                typer.echo(' ')
                typer.echo(f"📦 {p} → embedding[{len(arr)}]")
                typer.echo(arr)
            except Exception as e:
                typer.echo(f"❌ Error en {p}: {e}", err=True)