from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd

from multimodal_fin.financial.visualizer import EventVisualizer
from multimodal_fin.financial.data_loader import DataLoader
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EventResult:
    """Stores calculation results of an event study."""
    alpha: float
    beta: float
    df_results: pd.DataFrame  # ['Date','Return_stock','Return_market','Expected','AR','CAR','t']
    car_by_window: Dict[Tuple[int, int], float]  # e.g. {(0,30): 0.0123}
    n_estimation_obs: int


@dataclass
class Event:
    """Represents a single financial event (e.g. earnings call)."""

    event_date: datetime
    ticker: str
    company_name: Optional[str] = None
    quarter: Optional[str] = None
    year: Optional[int] = None

    estimation_start: Optional[datetime] = None
    estimation_end: Optional[datetime] = None
    gap: Optional[int] = None

    result: Optional[EventResult] = None  # se rellena tras cálculo

    def compute_car(self, t1: int, t2: int) -> Optional[float]:
        """Compute CAR on demand for a given event window."""
        if self.result is None or self.result.df_results is None:
            return None
        df = self.result.df_results
        dfw = df[(df["t"] >= t1) & (df["t"] <= t2)]
        return float(dfw["AR"].sum()) if not dfw.empty else None
        

    def report(self, plot_config: Optional[dict] = None) -> None:
        plot_config = plot_config or {"plot_car": True, "plot_periods": True, "plot_year": None}

        if self.result is None:
            print(f"⚠️ Event {self.ticker} on {self.event_date.date()} has no results.")
            return

        (t1, t2), car_val = next(iter(self.result.car_by_window.items()))

        print(f"\n📅 Event: {self.event_date.date()} ({self.ticker})")
        print(f"Company: {self.company_name or 'Unknown'}")
        print(f"Estimation window: {self.estimation_start.date()} → {self.estimation_end.date()} (gap={self.gap})")
        print(f"Alpha: {self.result.alpha:.6f}")
        print(f"Beta: {self.result.beta:.6f}")
        print(f"CAR [{t1},{t2}]: {car_val:.4%}")

        loader = DataLoader("/home/aacastro/2025_ACA_MultichannelAI/data/financial/companies_closes")
        df_stock = loader.load_returns(self.ticker)

        if plot_config.get("plot_car", True):
            EventVisualizer.plot_ar_car(self, t1, t2)

        if plot_config.get("plot_periods", True):
            EventVisualizer.plot_event_periods(
                df_returns=df_stock,
                ticker=self.ticker,
                event_date=self.event_date,
                estimation_start=self.estimation_start,
                estimation_end=self.estimation_end,
                event_start=self.event_date + pd.Timedelta(days=t1),
                event_end=self.event_date + pd.Timedelta(days=t2),
            )