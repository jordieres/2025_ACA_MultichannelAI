import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta


class EventVisualizer:
    """Handles event study visualizations."""

    @staticmethod
    def plot_ar_car(df: pd.DataFrame, event_date: str) -> None:
        """Plot abnormal returns (AR) and cumulative abnormal returns (CAR),
        highlighting the event date in red bold font, with full date labels.

        Args:
            df (pd.DataFrame): Data with AR and CAR.
            event_date (str): Event date string (YYYY-MM-DD).
        """
        df = df.copy()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(df["Date"], df["AR"], color="skyblue", label="AR diario")
        ax1.set_ylabel("Daily Abnoral Returns (AR)")
        ax1.axhline(0, color="gray", linestyle="--")

        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["CAR"], color="red", label="Cumulative Abnormal Returns (CAR)", linewidth=2)
        ax2.set_ylabel("Cumulative Abnormal Returns (CAR)")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        ax1.set_xlabel("Date")

        fechas = df["Date"].dt.strftime("%Y-%m-%d")
        event_date_str = pd.to_datetime(event_date).strftime("%Y-%m-%d")

        ax1.set_xticks(df["Date"])
        ax1.set_xticklabels(fechas, rotation=45, ha="right", fontsize=9)

        for label, fecha in zip(ax1.get_xticklabels(), fechas):
            if fecha == event_date_str:
                label.set_color("red")
                label.set_fontweight("bold")
            else:
                label.set_color("black")
                label.set_fontweight("normal")

        plt.title("Abnormal Returns (AR) and Cumulative Abnormal Returns (CAR)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_event_periods(
        df_returns: pd.DataFrame,
        ticker: str,
        event_date: str,
        estimation_start: str,
        estimation_end: str,
        event_start: str,
        event_end: str,
    ) -> None:
        """Plot estimation, gap and event periods with duration brackets.

        Args:
            df_returns (pd.DataFrame): DataFrame with columns ['Date', 'Return'].
            ticker (str): Stock ticker symbol.
            event_date (str): Event date (YYYY-MM-DD).
            estimation_start (str): Start of estimation window.
            estimation_end (str): End of estimation window.
            event_start (str): Start of event window.
            event_end (str): End of event window.
        """
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

        # Brackets with durations
        y_pos = df["Return"].max() * 1.1

        def draw_bracket(x1, x2, text, color):
            xm = x1 + (x2 - x1) / 2
            plt.hlines(y=y_pos, xmin=x1, xmax=x2, color=color, linewidth=2)
            plt.text(xm, y_pos + 0.002, text, color=color, fontsize=10, ha="center")

        draw_bracket(t_est_start, t_est_end, f"Estimation ({estimation_duration} days)", "blue")
        draw_bracket(
            t_est_end + timedelta(days=1),
            t_ev_start - timedelta(days=1),
            f"Gap ({gap_duration} days)",
            "gray",
        )
        draw_bracket(t_ev_start, t_ev_end, f"Event ({event_duration} days)", "darkred")

        plt.title(f"Key periods visualization of the event study ({ticker})")
        plt.xlabel("Date")
        plt.ylabel("Daily log return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()