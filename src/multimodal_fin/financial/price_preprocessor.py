import os
import pandas as pd
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class PricePreprocessor:
    """Prepares historical CSVs per ticker from a master price dataset."""

    def __init__(self, price_file: str, output_folder: str):
        """
        Args:
            price_file (str): Path to the master CSV with all tickers.
            output_folder (str): Folder where per-ticker files will be stored.
        """
        self.price_file = price_file
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def split_by_ticker(self) -> None:
        """Split master price CSV into individual per-ticker CSVs."""
        logger.info(f"Loading master price file: {self.price_file}")
        df = pd.read_csv(self.price_file, index_col=0)
        df["Date"] = pd.to_datetime(df["Date"])

        tickers = df["Ticker"].unique()
        logger.info(f"Found {len(tickers)} unique tickers.")

        for ticker in tickers:
            df_ticker = df[df["Ticker"] == ticker][["Date", "Close"]]
            df_ticker.sort_values("Date", inplace=True)

            filename = os.path.join(self.output_folder, f"{ticker}_historico.csv")
            df_ticker.to_csv(filename, index=False)
            logger.debug(f"Saved {ticker} data to {filename}")

        logger.info(f"Finished splitting dataset into {len(tickers)} CSV files.")

        return tickers
