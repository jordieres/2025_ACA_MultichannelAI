import os
import pandas as pd
import numpy as np
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Utility class to load and preprocess stock price data (one CSV per ticker)."""

    def __init__(self, folder_path: str):
        """
        Args:
            folder_path: Directory containing `<TICKER>_historico.csv` files.
        """
        self.folder_path = folder_path

    def load_returns(self, ticker: str) -> pd.DataFrame:
        """Load daily close CSV and compute log-returns.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            pd.DataFrame: Columns ['Date', 'Return'] sorted by Date.
        """
        path = os.path.join(self.folder_path, f"{ticker}_historico.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No CSV found for {ticker} at {path}")

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").copy()
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df[["Date", "Return"]].dropna()
