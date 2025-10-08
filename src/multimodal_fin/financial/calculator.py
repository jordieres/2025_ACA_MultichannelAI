from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm

from multimodal_fin.financial.data_loader import DataLoader
from multimodal_fin.financial.event import Event, EventResult
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)

class EventCalculator:
    """Market-model estimator and AR/CAR calculator for a given event."""

    def __init__(self, data_loader: DataLoader, market_ticker: str = "SPGI",
                 use_hac: bool = False, hac_lags: int = 5):
        self.loader = data_loader
        self.market_ticker = market_ticker
        self.use_hac = use_hac
        self.hac_lags = hac_lags

    def analyze_event(
        self,
        ev: Event,
        window_to_report: Tuple[int, int] = None
    ) -> Event:
        """Estimate (alpha, beta), compute AR/CAR, and fill ev.result.

        Args:
            ev: Event to analyze. Must have estimation_start/end set.
            window_to_report: Single (t1, t2) window to compute CAR.

        Returns:
            The same Event with `result` populated.
        """
        if ev.estimation_start is None or ev.estimation_end is None:
            raise ValueError(f"Event {ev.ticker} {ev.event_date.date()} has no estimation window assigned.")

        # Load returns
        df_stock = self.loader.load_returns(ev.ticker).rename(columns={"Return": "Return_stock"})
        df_mkt = self.loader.load_returns(self.market_ticker).rename(columns={"Return": "Return_market"})

        df = pd.merge(df_stock, df_mkt, on="Date", how="inner").dropna()
        df = df.sort_values("Date")

        # Estimation window
        df_est = df[(df["Date"] >= ev.estimation_start) & (df["Date"] <= ev.estimation_end)].copy()
        n_est = len(df_est)
        if n_est == 0:
            logger.error(f"[{ev.ticker}] No stock data in estimation window for event: {ev.quarter} {ev.year} ")
            ev.result = None
            return ev

        # OLS
        X = sm.add_constant(df_est["Return_market"])
        y = df_est["Return_stock"]
        model = sm.OLS(y, X)
        if self.use_hac:
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
        else:
            fit = model.fit()

        alpha = float(fit.params["const"])
        beta = float(fit.params["Return_market"])

        # Full AR/CAR
        df["Expected"] = alpha + beta * df["Return_market"]
        df["AR"] = df["Return_stock"] - df["Expected"]
        df["t"] = (df["Date"] - ev.event_date).dt.days
        df["CAR"] = df["AR"].cumsum()

        # Default or provided window
        if window_to_report is None:
            raise ValueError("A wondow to report must be provided.")
        elif not isinstance(window_to_report, tuple) or len(window_to_report) != 2:
            raise ValueError("window_to_report must be a tuple (t1, t2).")

        t1, t2 = window_to_report
        dfw = df[(df["t"] >= t1) & (df["t"] <= t2)]
        car_val = float(dfw["AR"].sum()) if not dfw.empty else np.nan

        car_by_window = {window_to_report: car_val}

        ev.result = EventResult(
            alpha=alpha,
            beta=beta,
            df_results=df,
            car_by_window=car_by_window,
            n_estimation_obs=n_est
        )
        return ev