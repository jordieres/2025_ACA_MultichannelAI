import os
import pandas as pd
import numpy as np

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)



class DataLoader:
    """Utility class to load and preprocess stock data."""

    def __init__(self, folder_path: str):
        """
        Args:
            folder_path (str): Path to the folder containing ticker CSVs.
        """
        self.folder_path = folder_path

    def load_returns(self, ticker: str) -> pd.DataFrame:
        """Load CSV for a ticker and compute log returns.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with Date and Return columns.
        """
        path = os.path.join(self.folder_path, f"{ticker}_historico.csv")
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"No CSV found for {ticker}")

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df[["Date", "Return"]].dropna()
