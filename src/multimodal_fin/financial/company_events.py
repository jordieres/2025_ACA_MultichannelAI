from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BDay

from multimodal_fin.financial.event import Event
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CompanyEvents:
    """Container and scheduler for all events of a single company."""

    ticker: str
    origin_companies_closes: str
    company_name: Optional[str] = None
    events: List[Event] = field(default_factory=list)

    def add_event(self, event_date: str | datetime,
                  quarter: Optional[str] = None,
                  year: Optional[int] = None) -> None:
        """Add event and keep them sorted by date."""
        ed = pd.to_datetime(event_date)

        self.events.append(
            Event(
                event_date=ed,
                ticker=self.ticker,
                company_name=self.company_name,
                quarter=quarter,
                year=year,
                origin_companies_closes=self.origin_companies_closes
            )
        )
        self.events.sort(key=lambda e: e.event_date)

    def assign_estimation_windows(
        self,
        *,
        est_len_days: int = 180,     # longitud máxima
        gap_days: int = 30,          # separación antes del evento
        prev_post_days: int = 7,    # días después del evento anterior
        min_obs_days: int = 30,      # longitud mínima
        window_to_report: Tuple[int, int] = (-7, 7),
    ) -> None:
        """Assign adaptive estimation windows avoiding contamination.

        Rules:
            event_start = t0 + t1
            estimation_end = min(event_start - gap_days,
                                 prev_event_date + prev_post_days - 1 if prev exists)
            estimation_start = estimation_end - est_len_days  # máx longitud
            if (estimation_end - estimation_start) < min_obs_days -> discard
        """
        if not self.events:
            return

        self.events.sort(key=lambda e: e.event_date)
        valid_events: List[Event] = []

        t1, _t2 = window_to_report

        for i, ev in enumerate(self.events):
            t0 = ev.event_date
            event_start = t0 + timedelta(days=t1)
            estimation_end = event_start - timedelta(days=gap_days)

            # limitar por evento anterior
            if i > 0:
                prev_t = self.events[i - 1].event_date
                end_cap = prev_t + timedelta(days=prev_post_days) - timedelta(days=1)
                estimation_end = min(estimation_end, end_cap)

            # ventana base de 180d, pero ajustable
            estimation_start = estimation_end - timedelta(days=est_len_days)

            # si la ventana invade el evento anterior, recortarla
            if i > 0:
                prev_event_start = self.events[i - 1].event_date + timedelta(days=t1)
                if estimation_start < prev_event_start:
                    estimation_start = prev_event_start

            actual_days = (estimation_end - estimation_start).days
            if actual_days < min_obs_days:
                logger.warning(
                    f"[{self.ticker}] Skipping {t0.date()}: "
                    f"{actual_days} days available (< {min_obs_days})"
                )
                continue

            ev.estimation_start = pd.Timestamp(estimation_start)
            ev.estimation_end = pd.Timestamp(estimation_end)
            ev.gap = gap_days
            valid_events.append(ev)

            logger.debug(
                f"[{self.ticker}] Event {t0.date()} window "
                f"{estimation_start.date()} → {estimation_end.date()} "
                f"({actual_days} days)"
            )

        self.events = valid_events

    def visualize_company_year(
        self,
        ticker: str,
        year: int,
        window_to_report: Tuple[int, int] = (-7, 7),
        show_estimation: bool = True,
        show_gap: bool = True,
        show_event: bool = True,
    ) -> None:
        """Visualize all event study periods for a company in a given year.

        Args:
            ticker: Stock ticker (must exist in the conference file).
            year: Year to visualize (e.g., 2024).
            window_to_report: Tuple (t1, t2) of event window.
            show_estimation, show_gap, show_event: Whether to show each region.
        """
        df_t = self.df_conf[self.df_conf["symbol"] == ticker]
        if df_t.empty:
            logger.warning(f"No events found for ticker {ticker}.")
            return

        # Construir CompanyEvents para ese ticker
        comp = self._build_company(ticker)

        # Si aún no tienen ventanas asignadas, las generamos
        if not any(e.estimation_start for e in comp.events):
            comp.assign_estimation_windows(
                est_len_days=180,
                gap_days=30,
                prev_post_days=60,
                min_obs_days=50,
                window_to_report=window_to_report,
            )

        # Cargar retornos de la acción
        df_stock = self.loader.load_returns(ticker)

        # Llamada al visualizador
        EventVisualizer.plot_company_year(
            company_events=comp,
            df_returns=df_stock,
            year=year,
            window_to_report=window_to_report,
            show_estimation=show_estimation,
            show_gap=show_gap,
            show_event=show_event,
        )