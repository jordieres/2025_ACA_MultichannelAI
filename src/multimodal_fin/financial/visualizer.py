import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multimodal_fin.financial.event import Event


class EventVisualizer:
    """Handles event study visualizations."""

    @staticmethod
    def plot_ar_car(event: "Event", t1: int, t2: int) -> None:
        """Plot abnormal returns (AR) and cumulative abnormal returns (CAR) 
        for a given event and time window.

        Args:
            event (Event): Event object with df_results.
            t1 (int): Start of event window (relative to event day).
            t2 (int): End of event window (relative to event day).
        """
        if event.result is None or event.result.df_results is None:
            print(f"No results available for {event.ticker} on {event.event_date.date()}")
            return

        df = event.result.df_results
        df = df[(df["t"] >= t1) & (df["t"] <= t2)].copy()
        if df.empty:
            print(f"No data available for window [{t1}, {t2}] in {event.ticker}")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Daily AR
        ax1.bar(df["t"], df["AR"], color="skyblue", label="Daily AR")
        ax1.set_ylabel("Daily Abnormal Returns (AR)")
        ax1.axhline(0, color="gray", linestyle="--")

        # Cumulative AR
        ax2 = ax1.twinx()
        ax2.plot(df["t"], df["CAR"], color="red", label="Cumulative AR", linewidth=2)
        ax2.set_ylabel("Cumulative Abnormal Returns (CAR)")

        # Legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        ax1.set_xlabel("Days relative to event (t)")
        plt.title(f"AR & CAR for {event.ticker} around {event.event_date.date()} [{t1},{t2}]")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_event_periods(
        df_returns: pd.DataFrame,
        ticker: str,
        event_date: pd.Timestamp,
        estimation_start: pd.Timestamp,
        estimation_end: pd.Timestamp,
        event_start: pd.Timestamp,
        event_end: pd.Timestamp,
    ) -> None:
        """Plot estimation, gap and event periods with duration brackets."""

        df = df_returns.copy()

        t_event = pd.to_datetime(event_date)
        t_est_start = pd.to_datetime(estimation_start)
        t_est_end = pd.to_datetime(estimation_end)
        t_ev_start = pd.to_datetime(event_start)
        t_ev_end = pd.to_datetime(event_end)

        estimation_duration = (t_est_end - t_est_start).days + 1
        gap_duration = (t_ev_start - t_est_end).days - 1
        event_duration = (t_ev_end - t_ev_start).days + 1

        # Clip range for better visualization
        df = df[
            (df["Date"] >= t_est_start - timedelta(days=5))
            & (df["Date"] <= t_ev_end + timedelta(days=5))
        ]

        plt.figure(figsize=(14, 6))
        plt.plot(df["Date"], df["Return"], color="black", alpha=0.5, label="Daily return")

        # Shaded periods
        plt.axvspan(t_est_start, t_est_end, color="skyblue", alpha=0.3, label="Estimation")
        plt.axvspan(t_est_end, t_ev_start, color="gray", alpha=0.1, label="Gap")
        plt.axvspan(t_ev_start, t_ev_end, color="salmon", alpha=0.3, label="Event")

        # Mark event day
        plt.axvline(t_event, color="red", linestyle="--", linewidth=1.2, label="Event day")

        # Brackets
        y_pos = df["Return"].max() * 1.1

        def draw_bracket(x1, x2, text, color):
            xm = x1 + (x2 - x1) / 2
            plt.hlines(y=y_pos, xmin=x1, xmax=x2, color=color, linewidth=2)
            plt.text(xm, y_pos + 0.002, text, color=color, fontsize=10, ha="center")

        draw_bracket(t_est_start, t_est_end, f"Estimation ({estimation_duration}d)", "blue")
        draw_bracket(
            t_est_end + timedelta(days=1),
            t_ev_start - timedelta(days=1),
            f"Gap ({gap_duration}d)",
            "gray",
        )
        draw_bracket(t_ev_start, t_ev_end, f"Event ({event_duration}d)", "darkred")

        plt.title(f"Key periods visualization of {ticker} ({event_date.date()})")
        plt.xlabel("Date")
        plt.ylabel("Daily log return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_company_year(
        company_events: "CompanyEvents",
        df_returns: pd.DataFrame,
        year: int | None = None,
        window_to_report: tuple[int, int] = (-7, 7),
        show_estimation: bool = True,
        show_gap: bool = True,
        show_event: bool = True,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> None:
        """Plot all events for a company (timeline style).

        Works either for a specific year or for a custom date range if provided.

        Args:
            company_events: CompanyEvents object containing all events.
            df_returns: DataFrame with columns ['Date', 'Return'].
            year: Year to visualize (optional if start/end dates provided).
            window_to_report: Tuple (t1, t2) of the event window.
            show_estimation, show_gap, show_event: Whether to display each region.
            start_date: Start date for the visualization (optional).
            end_date: End date for the visualization (optional).
        """
        # 🧠 Selección de eventos
        if year is not None:
            events = [e for e in company_events.events if e.event_date.year == year]
        else:
            events = [
                e
                for e in company_events.events
                if (start_date is None or e.event_date >= start_date)
                and (end_date is None or e.event_date <= end_date)
            ]

        if not events:
            period_label = f"{year}" if year else f"{start_date.date()} → {end_date.date()}"
            print(f"No events for {company_events.ticker} in {period_label}.")
            return

        df = df_returns.copy()
        plt.figure(figsize=(14, 3))
        plt.plot(df["Date"], df["Return"], color="black", alpha=0.3, label="Daily return")

        # base y-levels
        base_y = 0.02
        height = 0.02

        for i, ev in enumerate(events):
            color_est = "skyblue"
            color_gap = "gray"
            color_ev = "salmon"

            t1, t2 = window_to_report
            event_start = ev.event_date + pd.Timedelta(days=t1)
            event_end = ev.event_date + pd.Timedelta(days=t2)
            y_offset = base_y + i * (height * 2)

            if show_estimation and ev.estimation_start and ev.estimation_end:
                plt.axvspan(ev.estimation_start, ev.estimation_end, color=color_est, alpha=0.4)
                # plt.text(ev.estimation_start, y_offset, f"{ev.event_date.date()} est.", color=color_est, fontsize=8)

            if show_gap and ev.estimation_end:
                gap_start = ev.estimation_end
                gap_end = event_start
                plt.axvspan(gap_start, gap_end, color=color_gap, alpha=0.2)

            if show_event:
                plt.axvspan(event_start, event_end, color=color_ev, alpha=0.4)
                plt.axvline(ev.event_date, color="red", linestyle="--", linewidth=1)

        # 🏷️ Título adaptativo
        if year is not None:
            title_period = f"{year}"
        elif start_date and end_date:
            title_period = f"{start_date.date()} → {end_date.date()}"
        else:
            title_period = "full period"

        plt.title(f"{company_events.ticker} — Event study periods ({title_period})")
        plt.xlabel("Date")
        plt.ylabel("Return (normalized)")
        plt.legend(["Daily return", "Estimation", "Gap", "Event"], loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()