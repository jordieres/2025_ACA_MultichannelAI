import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta


class EventCalculator:
    """Performs market model estimation and abnormal return calculations."""

    def __init__(self, event_date: str, t1_offset: int = -7, t2_offset: int = 7):
        self.event_date = datetime.strptime(event_date, "%Y-%m-%d")
        self.t1_offset = t1_offset
        self.t2_offset = t2_offset
        self.alpha = None
        self.beta = None

    def get_windows(self) -> dict:
        """Define estimation and event windows.

        Returns:
            dict: Dictionary with estimation and event start/end dates.
        """
        return {
            "estimation_start": self.event_date - timedelta(days=80),
            "estimation_end": self.event_date - timedelta(days=27),
            "event_start": self.event_date + timedelta(days=self.t1_offset),
            "event_end": self.event_date + timedelta(days=self.t2_offset),
        }

    def estimate_market_model(self, df_stock: pd.DataFrame, df_market: pd.DataFrame) -> None:
        """Estimate alpha and beta using market model.

        Args:
            df_stock (pd.DataFrame): Stock returns (Date, Return).
            df_market (pd.DataFrame): Market returns (Date, Return).
        """
        df = pd.merge(df_stock, df_market, on="Date", suffixes=("_stock", "_market"))
        windows = self.get_windows()
        df_window = df[
            (df["Date"] >= windows["estimation_start"]) & (df["Date"] <= windows["estimation_end"])
        ].dropna()

        X = sm.add_constant(df_window["Return_market"])
        y = df_window["Return_stock"]
        model = sm.OLS(y, X).fit()
        self.alpha, self.beta = model.params

    def calculate_abnormal_returns(
        self, df_stock: pd.DataFrame, df_market: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate abnormal returns (AR) and cumulative abnormal returns (CAR).

        Args:
            df_stock (pd.DataFrame): Stock returns.
            df_market (pd.DataFrame): Market returns.

        Returns:
            pd.DataFrame: DataFrame with AR and CAR values.
        """
        if self.alpha is None or self.beta is None:
            raise ValueError("Alpha and beta must be estimated before calculating AR.")

        df = pd.merge(df_stock, df_market, on="Date", suffixes=("_stock", "_market"))
        windows = self.get_windows()
        df_event = df[
            (df["Date"] >= windows["event_start"]) & (df["Date"] <= windows["event_end"])
        ].dropna()

        df_event["Expected"] = self.alpha + self.beta * df_event["Return_market"]
        df_event["AR"] = df_event["Return_stock"] - df_event["Expected"]
        df_event["CAR"] = df_event["AR"].cumsum()

        return df_event
