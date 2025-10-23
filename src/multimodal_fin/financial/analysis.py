import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import List, Tuple
from multimodal_fin.financial.event import Event
from multimodal_fin.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_statistical_significance(
    events: List[Event], car_window: Tuple[int, int] = (0, 1)
) -> pd.DataFrame:
    """Analyze statistical significance of AR and CAR across multiple events.

    Args:
        events (List[Event]): List of Event objects.
        car_window (Tuple[int, int], optional): Relative days for CAR calculation. Defaults to (0, 1).

    Returns:
        pd.DataFrame: DataFrame with average AR statistics by t, plus CAR test results.
    """
    ar_dict = {}
    car_list = []

    for event in events:
        df = event.df_results.copy()
        df["t"] = (df["Date"] - pd.to_datetime(event.event_date)).dt.days

        # Save ARs per t
        for _, row in df.iterrows():
            t = row["t"]
            ar_dict.setdefault(t, []).append(row["AR"])

        # CAR in sub-window
        df_window = df[(df["t"] >= car_window[0]) & (df["t"] <= car_window[1])]
        car = df_window["AR"].sum()
        car_list.append(car)

    # Compute average AR stats per t
    stats_ar = {
        t: {
            "mean_AR": np.mean(ars),
            "std_AR": np.std(ars, ddof=1),
            "t_stat": stats.ttest_1samp(ars, 0.0)[0],
            "p_value": stats.ttest_1samp(ars, 0.0)[1],
            "n": len(ars),
        }
        for t, ars in sorted(ar_dict.items())
    }

    df_ar_stats = (
        pd.DataFrame.from_dict(stats_ar, orient="index")
        .reset_index()
        .rename(columns={"index": "t"})
    )

    # CAR test
    tstat_car, pval_car = stats.ttest_1samp(car_list, 0.0)
    car_mean = np.mean(car_list)
    car_std = np.std(car_list, ddof=1)

    logger.info(
        f"CAR mean in window {car_window}: {car_mean:.4%} "
        f"(t-stat={tstat_car:.3f}, p-value={pval_car:.5f}, n_samples={len(car_list)})"
    )

    # Add CAR summary as extra row in df
    summary = pd.DataFrame(
        {
            "t": ["CAR_window"],
            "mean_AR": [car_mean],
            "std_AR": [car_std],
            "t_stat": [tstat_car],
            "p_value": [pval_car],
            "n": [len(car_list)],
        }
    )

    df_result = pd.concat([df_ar_stats, summary], ignore_index=True)
    return df_result, summary


def plot_AR_significance(df_ar_stats: pd.DataFrame, alpha: float = 0.05, window: tuple[int, int] = None) -> None:
    """Plot average AR per relative day (t) with error bars and significance markers.

    Args:
        df_ar_stats (pd.DataFrame): DataFrame with columns
            ['t', 'mean_AR', 'std_AR', 'n', 't_stat', 'p_value'].
        alpha (float, optional): Significance threshold. Defaults to 0.05.
        window (tuple[int,int], optional): (t1, t2) event window to highlight. 
            If provided, the x-axis will be clipped to this range.
    """
    # Filtrar solo filas con t numérico
    df = df_ar_stats[pd.to_numeric(df_ar_stats["t"], errors="coerce").notna()].copy()
    df["t"] = df["t"].astype(int)

    plt.figure(figsize=(12, 6))

    # Standard error
    df["error_std"] = df["std_AR"] / np.sqrt(df["n"])

    # Plot mean AR with error bars
    plt.errorbar(
        df["t"],
        df["mean_AR"],
        yerr=df["error_std"],
        fmt="o",
        capsize=5,
        label="Mean AR ± Std. Error",
        color="blue",
    )

    # Mark significant points
    for _, row in df.iterrows():
        if row["p_value"] < alpha:
            plt.text(
                row["t"],
                row["mean_AR"] + 1.5 * row["error_std"],
                "*",
                ha="center",
                va="bottom",
                color="red",
                fontsize=15,
            )

    # Reference line
    plt.axhline(0, color="gray", linestyle="--")

    # Highlight window if provided
    if window is not None:
        plt.axvspan(window[0], window[1], color="orange", alpha=0.1, label=f"Window {window}")

    plt.title("Average Abnormal Returns (AR) by day relative to event")
    plt.xlabel("Relative day (t)")
    plt.ylabel("Average Abnormal Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()