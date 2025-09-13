"""
Runner for downloading conference data (transcripts and audio) from earningscall.biz.

It uses a subset of S&P500 companies grouped by sector.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import earningscall

from multimodal_fin.config import DataAdquisitionSettings
from multimodal_fin.data_adquisition.Company import CompanyDataAcquisition
from multimodal_fin.runners.base import Runner
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class DataAdquisitionRunner(Runner):
    """Runner responsible for fetching earnings call transcripts and audio."""

    def __init__(self, settings: DataAdquisitionSettings):
        """Initialize the data acquisition runner.

        Args:
            settings (DataAdquisitionSettings): Configuration with API key, base path, and URL.
        """
        self.settings = settings

    def run(self, **kwargs) -> None:
        """Download data for S&P500 companies from earningscall.biz.

        This includes scraping the main page, parsing the company table,
        and triggering download for transcripts and audio files.
        """
        logger.info("Starting data acquisition from earningscall.biz")
        earningscall.api_key = self.settings.api_key

        try:
            response = requests.get(self.settings.url)
            response.raise_for_status()
            logger.info(f"Successfully fetched data from URL: {self.settings.url}")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data from {self.settings.url}: {e}", exc_info=True)
            raise

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        if not table:
            logger.error("No table found on the earningscall page.")
            raise ValueError("HTML table not found")

        headers = [header.text.strip() for header in table.find_all('th')]
        rows = [
            [col.text.strip() for col in row.find_all('td')]
            for row in table.find_all('tr')[1:]
        ]
        df = pd.DataFrame(rows, columns=headers)
        logger.info(f"Parsed table with {len(df)} entries.")

        # Select top 8 companies per sector for demonstration purposes
        sp500_subset = df.groupby("Sector").head(8).reset_index(drop=True)
        logger.info(f"Selected {len(sp500_subset)} companies (top 8 per sector)")

        for code in sp500_subset['Symbol']:
            logger.info(f"Fetching data for company: {code}")
            try:
                company = CompanyDataAcquisition(code)
                company.get_and_save_all_transcripts_and_audio(self.settings.base_path)
                logger.info(f"Successfully downloaded data for {code}")
            except Exception as e:
                logger.error(f"Failed to download data for {code}: {e}", exc_info=True)