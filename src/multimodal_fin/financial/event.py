from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from multimodal_fin.financial.data_loader import DataLoader
from multimodal_fin.financial.calculator import EventCalculator
from multimodal_fin.financial.visualizer import EventVisualizer

from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Event:
    """Domain class representing a financial event."""

    event_date: str
    ticker: str
    market_ticker: str = "SPGI"
    folder_path: str = "./data"
    t1_offset: int = -7
    t2_offset: int = 7
    plot_flags: Optional[dict] = None

    df_results: pd.DataFrame = field(init=False)
    car: float = field(init=False)
    alpha: float = field(init=False)
    beta: float = field(init=False)

    def __post_init__(self):
        self.loader = DataLoader(self.folder_path)
        self.calculator = EventCalculator(self.event_date, self.t1_offset, self.t2_offset, self.ticker)

        df_stock = self.loader.load_returns(self.ticker)
        df_market = self.loader.load_returns(self.market_ticker)

        try:
            self.calculator.estimate_market_model(df_stock, df_market)
            df_event = self.calculator.calculate_abnormal_returns(df_stock, df_market)
        except ValueError as e:
            logger.error(f"[Event] Skipping {self.ticker} on {self.event_date}: {e}")
            self.df_results = None
            self.car = None
            return


        self.df_results = df_event
        self.car = df_event["AR"].sum()

        self.alpha = self.calculator.alpha
        self.beta = self.calculator.beta

        logger.info(f"Finished event study for {self.ticker} | CAR = {self.car:.4%}")

    def summary(self) -> None:
        """Print event summary."""
        print(f"\n📅 Event: {self.event_date} ({self.ticker})")
        print(f"Alpha: {self.alpha:.6f}")
        print(f"Beta: {self.beta:.6f}")
        print(f"CAR [{self.t1_offset}, {self.t2_offset}]: {self.car:.4%}")

        if self.plot_flags and any(self.plot_flags.values()):
            if self.plot_flags.get("car", False):
                EventVisualizer.plot_ar_car(self.df_results, self.event_date)

            if self.plot_flags.get("returns", False):
                df_stock = self.loader.load_returns(self.ticker)
                EventVisualizer.plot_event_periods(
                    df_returns=df_stock,
                    ticker=self.ticker,
                    event_date=self.event_date,
                    estimation_start=self.calculator.get_windows()["estimation_start"],
                    estimation_end=self.calculator.get_windows()["estimation_end"],
                    event_start=self.calculator.get_windows()["event_start"],
                    event_end=self.calculator.get_windows()["event_end"],
                )

            if self.plot_flags.get("modelo", False):
                # TODO: implementar plot de modelo de mercado
                logger.info("Plotting market model (not yet implemented).")


