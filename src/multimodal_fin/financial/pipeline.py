import pandas as pd
import os
from typing import Optional, List, Tuple, Union
from rich.console import Console
from rich.table import Table
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from multimodal_fin.financial.data_loader import DataLoader
from multimodal_fin.financial.calculator import EventCalculator
from multimodal_fin.financial.company_events import CompanyEvents
from multimodal_fin.financial.visualizer import EventVisualizer
from multimodal_fin.financial.event import Event
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)



class EventPipeline:
    """Loads conference CSV, builds companies with events, assigns windows, and analyzes events."""

    def __init__(self, conference_file: str, prices_folder: str, market_ticker: str = "SPGI"):
        """
        Args:
            conference_file: CSV with columns at least ['symbol','company','timestamp'].
            prices_folder: Folder where `<TICKER>_historico.csv` live.
            market_ticker: Benchmark ticker for the market model.
        """
        self.df_conf = pd.read_csv(conference_file, index_col=0)
        self.df_conf["timestamp"] = pd.to_datetime(self.df_conf["timestamp"])
        self.prices_folder = prices_folder
        self.market_ticker = market_ticker
        self.loader = DataLoader(prices_folder)
        self.calculator = EventCalculator(self.loader, market_ticker=self.market_ticker)

    def _build_company(self, ticker: str) -> CompanyEvents:
        """Create a CompanyEvents for a given ticker from the CSV."""
        df_t = self.df_conf[self.df_conf["symbol"] == ticker].sort_values("timestamp")
        if df_t.empty:
            raise ValueError(f"No events for ticker {ticker} in conference CSV.")

        company_name = df_t["company"].iloc[0] if "company" in df_t.columns else None
        comp = CompanyEvents(ticker=ticker, company_name=company_name, origin_companies_closes=self.prices_folder)
        for _, row in df_t.iterrows():
            comp.add_event(event_date=row["timestamp"], quarter=row["quarter"], year=row["year"])
        return comp

    def run(
        self,
        ticker_filter: Optional[Union[str, List[str]]] = None,
        *,
        est_len_days: int = 180,
        gap_days: int = 30,
        prev_post_days: int = 60,
        window_to_report: Tuple[int, int] = (-7, 7),
    ) -> List[Event]:
        """Process events (single ticker, multiple tickers, or all).

        Args:
            ticker_filter: Optional ticker or list of tickers to process.
            est_len_days: Estimation length in business days.
            gap_days: Gap before event in business days.
            prev_post_days: Excluded post-window after previous event.
            window_to_report: (t1, t2) window for CAR computation.

        Returns:
            List[Event]: Each with `result` filled (alpha, beta, df_results, car_by_window).
        """

        if ticker_filter is None:
            tickers = sorted(self.df_conf["symbol"].unique())
        elif isinstance(ticker_filter, str):
            tickers = [ticker_filter]
        elif isinstance(ticker_filter, list):
            tickers = list(ticker_filter)
        else:
            raise TypeError("ticker_filter must be a str, list or None.")
        all_events: List[Event] = []

        if not isinstance(window_to_report, tuple):
            raise ValueError("windows_to_report must be a tuple (t1, t2).")

        for tk in tickers:
            comp = self._build_company(tk)
            comp.assign_estimation_windows(
                est_len_days=est_len_days,
                gap_days=gap_days,
                prev_post_days=prev_post_days,
            )

            for ev in comp.events:
                try:
                    self.calculator.analyze_event(ev, window_to_report=window_to_report)
                except Exception as e:
                    logger.error(f"Error analyzing {tk} {ev.event_date.date()}: {e}")
                all_events.append(ev)

        return all_events
    
    def get_event(
        self,
        ticker: str,
        *,
        year: int,
        quarter: str,
        events: Optional[List[Event]] = None
    ) -> Event:
        """Retrieve a specific event by year and quarter for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g. "AMZN").
            year: Target year.
            quarter: Target quarter label (e.g. "Q1", "Q2", "Q3", "Q4").
            events: Optional list of precomputed events. If None, rebuilds them.

        Returns:
            Event: Matching event instance.

        Raises:
            ValueError: If no matching event is found.
        """
        if events is None:
            # If not provided, rebuild company and events
            comp = self._build_company(ticker)
            comp.assign_estimation_windows()
            events = comp.events

        matches = [e for e in events if e.ticker == ticker and e.year == year and e.quarter == quarter]
        if not matches:
            raise ValueError(f"No event found for {ticker} {year} {quarter}.")
        if len(matches) > 1:
            logger.warning(f"[{ticker}] Multiple matches for {year} {quarter}, returning the first.")
        return matches[0]
    
    def list_company_events(self, ticker: str, use_rich: bool = True, return_df: bool = False, verbose: int = 0):
        """List all events for a given company with quarter, year, and event date.

        Args:
            ticker (str): Ticker symbol of the company.
            use_rich (bool): Whether to use rich for styled console output.
            return_df (bool): If True, return a DataFrame with event info.

        Returns:
            Optional[pd.DataFrame]: Table with event info if return_df=True.
        """
        df_t = self.df_conf[self.df_conf["symbol"] == ticker].sort_values("timestamp")

        if df_t.empty:
            print(f"No events found for ticker '{ticker}'.")
            return None

        df_t = df_t[["timestamp", "quarter", "year", "label_start_date", "label_end_date"]]
        df_t["timestamp"] = pd.to_datetime(df_t["timestamp"])
        df_t["label_start_date"] = pd.to_datetime(df_t["label_start_date"], errors="coerce")
        df_t["label_end_date"] = pd.to_datetime(df_t["label_end_date"], errors="coerce")

        if verbose >= 1:
            if use_rich:
                console = Console()
                company_name = (
                    self.df_conf[self.df_conf["symbol"] == ticker]["company"].iloc[0]
                    if "company" in self.df_conf.columns
                    else ticker
                )
                table = Table(title=f"📊 Events for {company_name} ({ticker})", show_lines=True)

                # table.add_column("Index", justify="center", style="bold cyan")
                table.add_column("Date", justify="center")
                table.add_column("Quarter", justify="center")
                table.add_column("Year", justify="center")
                table.add_column("Event Window", justify="center", style="dim")

                for i, row in df_t.iterrows():
                    start = row["label_start_date"].date() if pd.notnull(row["label_start_date"]) else "—"
                    end = row["label_end_date"].date() if pd.notnull(row["label_end_date"]) else "—"
                    table.add_row(
                        # str(i),
                        str(row["timestamp"].date()),
                        row["quarter"] or "?",
                        str(row["year"]),
                        f"{start} → {end}",
                    )

                console.print(table)
            else:
                print(f"\nEvents for {ticker}:")
                print("-" * 60)
                for i, row in df_t.iterrows():
                    print(
                        f"[{i}] {row['timestamp'].date()} | {row['quarter']} {row['year']} "
                        f"| Window: {row['label_start_date']} → {row['label_end_date']}"
                    )
                print("-" * 60)

        if return_df:
            return df_t.reset_index(drop=True)
        
    def analyze_significance(
        self,
        ticker: Optional[str] = None,
        *,
        events: Optional[List[Event]] = None,
        car_window: Tuple[int, int] = (0, 1),
        plot: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze statistical significance of AR and CAR across events.

        Args:
            ticker (Optional[str]): Company ticker. If None, analyze all events provided.
            events (Optional[List[Event]]): List of events to analyze.
                If None, events are rebuilt for the given ticker.
            car_window (Tuple[int, int]): Relative days for CAR aggregation.
            plot (bool): Whether to show average AR plot across events.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - df_car: Individual CARs for each event
                - summary: Significance summary per ticker
        """
        if events is None:
            if ticker is None:
                raise ValueError("Must provide either `ticker` or `events`.")

            comp = self._build_company(ticker)
            comp.assign_estimation_windows()

            for ev in comp.events:
                try:
                    self.calculator.analyze_event(ev, window_to_report=car_window)
                except Exception as e:
                    logger.error(f"[{ticker}] Error analyzing event {ev.event_date.date()}: {e}")

            events = comp.events

        # ✅ Filtrar solo los eventos válidos
        events = [e for e in events if e.result is not None and e.result.car_by_window]

        if not events:
            raise ValueError("No valid events with CAR available for statistical analysis.")

        # ✅ Detectar tickers únicos en la lista de eventos
        tickers = sorted(set(e.ticker for e in events))

        # Si hay más de un ticker, no se hace plot general
        if len(tickers) > 1:
            plot = False
            logger.info(f"Multiple tickers detected: {tickers}. Plot disabled.")

        all_records = []
        all_summaries = []

        for tck in tickers:
            # Filtrar eventos de esa empresa
            tck_events = [e for e in events if e.ticker == tck]
            records = []

            for e in tck_events:
                car_val = e.result.car_by_window.get(car_window)
                records.append({
                    "ticker": e.ticker,
                    "year": e.year,
                    "quarter": e.quarter,
                    "event_date": e.event_date,
                    "CAR": car_val
                })

            df_car_tck = pd.DataFrame(records)

            # Estadísticas
            tstat_car, pval_car = stats.ttest_1samp(df_car_tck["CAR"], 0.0)
            car_mean = np.mean(df_car_tck["CAR"])
            car_std = np.std(df_car_tck["CAR"], ddof=1)

            logger.info(
                f"[{tck}] CAR mean in window {car_window}: {car_mean:.4%} "
                f"(t-stat={tstat_car:.3f}, p-value={pval_car:.5f}, n_samples={len(df_car_tck)})"
            )

            summary_tck = pd.DataFrame({
                "ticker": [tck],
                "car_window": [str(car_window)],
                "mean_CAR": [car_mean],
                "std_CAR": [car_std],
                "t_stat": [tstat_car],
                "p_value": [pval_car],
                "n_events": [len(df_car_tck)],
            })

            all_records.append(df_car_tck)
            all_summaries.append(summary_tck)

            # Plot individual solo si hay un ticker
            if plot:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4))
                plt.bar(
                    df_car_tck["quarter"].astype(str) + " " + df_car_tck["year"].astype(str),
                    df_car_tck["CAR"],
                    color="steelblue"
                )
                plt.axhline(0, color="black", linestyle="--", linewidth=1)
                plt.title(f"CARs by Event for {tck}")
                plt.xlabel("Quarter-Year")
                plt.ylabel("CAR")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        # ✅ Concatenar resultados de todas las empresas
        df_car = pd.concat(all_records, ignore_index=True)
        summary = pd.concat(all_summaries, ignore_index=True)

        return df_car, summary


    def plot_stock_conferences_evolution(
        self,
        ticker: str,
        *,
        start_date: Optional[str | pd.Timestamp] = None,
        end_date: Optional[str | pd.Timestamp] = None,
        show_quarters: bool = True,
        figsize: Tuple[int, int] = (14, 6),
        color_price: str = "black",
        color_event: str = "red",
    ) -> None:
        """Plot stock price evolution with conference call dates marked.

        This visualization shows the closing price of a company's stock and marks
        all its earnings/conference events as vertical lines.

        Args:
            ticker (str): Company ticker symbol (e.g., 'TSLA').
            start_date (Optional[str | pd.Timestamp]): Start of the visualization period.
                If None, starts from earliest available data.
            end_date (Optional[str | pd.Timestamp]): End of the visualization period.
                If None, ends at the latest available data.
            show_quarters (bool): Whether to annotate each conference line with its quarter.
            figsize (Tuple[int, int]): Figure size.
            color_price (str): Line color for stock price.
            color_event (str): Color for conference markers.

        Raises:
            ValueError: If the ticker is not present in the conference file or has no price data.
        """
        # --- Validations ---
        if ticker not in self.df_conf["symbol"].unique():
            raise ValueError(f"Ticker '{ticker}' not found in the conference data.")

        # --- Load stock price ---
        try:
            df_price = pd.read_csv(os.path.join(self.prices_folder, f'{ticker}_historico.csv'), parse_dates=["Date"]).sort_values("Date")   
        except FileNotFoundError:
            raise ValueError(f"No price data found for ticker '{ticker}' in {self.prices_folder}.")

        df_price = df_price.sort_values("Date")

        # Apply date filtering
        if start_date:
            df_price = df_price[df_price["Date"] >= pd.Timestamp(start_date)]
        if end_date:
            df_price = df_price[df_price["Date"] <= pd.Timestamp(end_date)]

        # --- Load conferences for this ticker ---
        df_conf_tk = self.df_conf[self.df_conf["symbol"] == ticker].sort_values("timestamp")

        if df_conf_tk.empty:
            logger.warning(f"No conferences found for {ticker}.")
            return

        # Apply date filtering also to conferences
        if start_date:
            df_conf_tk = df_conf_tk[df_conf_tk["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df_conf_tk = df_conf_tk[df_conf_tk["timestamp"] <= pd.Timestamp(end_date)]

        # --- Plot ---
        plt.figure(figsize=figsize)
        plt.plot(df_price["Date"], df_price["Close"], label=f"{ticker} Close", color=color_price, linewidth=1.8)

        for _, row in df_conf_tk.iterrows():
            ts = row["timestamp"]
            plt.axvline(ts, color=color_event, linestyle="--", alpha=0.7, linewidth=1.2)
            if show_quarters and "quarter" in row and not pd.isna(row["quarter"]):
                plt.text(
                    ts,
                    df_price["Close"].max() * 0.98,
                    f"{row['quarter']}",
                    color='black',
                    fontsize=12,
                    ha="center",
                    rotation=45,
                    alpha=0.7,
                )

        # --- Styling ---
        plt.title(f"{ticker} stock price with conference calls marked", fontsize=13)
        if start_date or end_date:
            plt.suptitle(f"Period: {start_date or df_price['Date'].min().date()} → {end_date or df_price['Date'].max().date()}", fontsize=10, y=0.94)
        plt.xlabel("Date")
        plt.ylabel("Close price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        logger.info(
            f"[{ticker}] Plotted stock evolution with {len(df_conf_tk)} conference markers "
            f"from {df_price['Date'].min().date()} to {df_price['Date'].max().date()}."
        )
    
    # def visualize_company_period(
    #     self,
    #     ticker: str,
    #     start_date: str | pd.Timestamp,
    #     end_date: str | pd.Timestamp,
    #     window_to_report: Tuple[int, int] = (-7, 7),
    #     show_estimation: bool = True,
    #     show_gap: bool = True,
    #     show_event: bool = True,
    # ) -> None:
    #     """Visualize event study periods for a company within a custom date range.

    #     Displays estimation, gap, and event windows for all events occurring within
    #     [start_date, end_date]. The return series will be clipped to that range for clarity.

    #     Args:
    #         ticker: Stock ticker (must exist in the conference file).
    #         start_date: Start of the visualization period.
    #         end_date: End of the visualization period.
    #         window_to_report: Tuple (t1, t2) of event window.
    #         show_estimation, show_gap, show_event: Whether to show each region.
    #     """
    #     start_date = pd.Timestamp(start_date)
    #     end_date = pd.Timestamp(end_date)

    #     # Filtrar conferencias de ese ticker
    #     df_t = self.df_conf[self.df_conf["symbol"] == ticker]
    #     if df_t.empty:
    #         logger.warning(f"No events found for ticker {ticker}.")
    #         return

    #     # Crear objeto CompanyEvents
    #     comp = self._build_company(ticker)

    #     # Si no tienen ventanas asignadas, las calculamos
    #     if not any(e.estimation_start for e in comp.events):
    #         comp.assign_estimation_windows(
    #             est_len_days=180,
    #             gap_days=30,
    #             prev_post_days=60,
    #             min_obs_days=50,
    #             window_to_report=window_to_report,
    #         )

    #     # Cargar retornos de la acción
    #     df_stock = self.loader.load_returns(ticker)

    #     # 🔧 Filtrar rango indicado
    #     df_stock = df_stock[(df_stock["Date"] >= start_date) & (df_stock["Date"] <= end_date)].copy()

    #     # ⚙️ Filtrar eventos dentro del rango (los que ocurren dentro de la ventana principal)
    #     comp.events = [e for e in comp.events if start_date <= e.event_date <= end_date]

    #     if not comp.events:
    #         logger.warning(f"[{ticker}] No events found within {start_date.date()} → {end_date.date()}.")
    #         return

    #     # 🪶 Llamada al visualizador
    #     EventVisualizer.plot_company_year(
    #         company_events=comp,
    #         df_returns=df_stock,
    #         year=None,  # ya no se usa, pero el visualizador lo ignora si no está
    #         window_to_report=window_to_report,
    #         show_estimation=show_estimation,
    #         show_gap=show_gap,
    #         show_event=show_event,
    #     )

    #     logger.info(
    #         f"[{ticker}] Visualized period {start_date.date()} → {end_date.date()} "
    #         f"with {len(comp.events)} events."
    #     )