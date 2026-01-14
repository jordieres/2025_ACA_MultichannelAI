import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from multimodal_fin.analysis_qa_effects.config import PipelineConfig

class AggregatedEffectPlotter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def forest_plot_aggregated_combined(self, df_all: pd.DataFrame):
        """
        One figure: mean g + CI for each emotion, two modalities side-by-side (audio/text).
        CI is for the mean across companies (normal approx).
        """
        canonical = list(self.config.emotions_order)
        emotions = [e for e in canonical if e in df_all["emotion"].unique()]

        n_companies = df_all["company"].nunique()
        alpha = 1 - self.config.ci_level
        z = norm.ppf(1 - alpha / 2)

        y = np.arange(len(emotions))
        fig, ax = plt.subplots(figsize=(8, 0.7 * len(emotions) + 2))

        colors = {
            "audio": plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            "text":  plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        }

        offset = 0.10

        for modality, dy in zip(["audio", "text"], [-offset, +offset]):
            means, lows, highs = [], [], []

            for emo in emotions:
                vals = df_all[
                    (df_all["emotion"] == emo) &
                    (df_all["modality"] == modality)
                ]["hedges_g"].dropna().values

                if len(vals) < self.config.min_n:
                    means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                    continue

                mean = float(np.mean(vals))
                se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ci_low = mean - z * se
                ci_high = mean + z * se

                means.append(mean); lows.append(ci_low); highs.append(ci_high)

            ax.hlines(y + dy, lows, highs, color=colors[modality], linewidth=2, alpha=0.8)
            ax.plot(means, y + dy, "o", color=colors[modality], label=modality.capitalize())

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(emotions)
        ax.set_xlabel("Mean Hedges g (Yes − Evasive)")
        ax.set_ylabel("Emotion")
        ax.set_title(f"Aggregated population-level effects (n = {n_companies} companies)")

        ax.legend()
        plt.tight_layout()
        plt.show()

    def grouped_violin_by_emotion(self, df_all: pd.DataFrame, offset=0.18, width=0.30):
        """
        One figure:
        - X axis: emotions
        - For each emotion: two violins (audio vs text)
        - Y axis: effect size (Hedges g)
        """
        canonical = list(self.config.emotions_order)
        emotions = [e for e in canonical if e in df_all["emotion"].unique()]
        if not emotions:
            emotions = sorted(df_all["emotion"].unique())

        audio_data, text_data = [], []
        for emo in emotions:
            a = df_all[(df_all["emotion"] == emo) & (df_all["modality"] == "audio")]["hedges_g"].dropna().values
            t = df_all[(df_all["emotion"] == emo) & (df_all["modality"] == "text")]["hedges_g"].dropna().values
            audio_data.append(a)
            text_data.append(t)

        x = np.arange(len(emotions)) + 1
        pos_audio = x - offset
        pos_text = x + offset

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_audio = color_cycle[0]
        c_text = color_cycle[1] if len(color_cycle) > 1 else color_cycle[0]

        fig, ax = plt.subplots(figsize=(1.25 * len(emotions) + 4, 5))

        v1 = ax.violinplot(audio_data, positions=pos_audio, widths=width,
                           showmeans=False, showmedians=True, showextrema=False)
        for body in v1["bodies"]:
            body.set_alpha(0.55)
            body.set_facecolor(c_audio)
            body.set_edgecolor(c_audio)
        v1["cmedians"].set_color(c_audio)
        v1["cmedians"].set_linewidth(2)

        v2 = ax.violinplot(text_data, positions=pos_text, widths=width,
                           showmeans=False, showmedians=True, showextrema=False)
        for body in v2["bodies"]:
            body.set_alpha(0.55)
            body.set_facecolor(c_text)
            body.set_edgecolor(c_text)
        v2["cmedians"].set_color(c_text)
        v2["cmedians"].set_linewidth(2)

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(emotions)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Hedges g (Yes − Evasive)")
        ax.set_title("Effect size distributions by emotion (Audio vs Text)")

        ax.legend(handles=[
            Patch(facecolor=c_audio, edgecolor=c_audio, alpha=0.55, label="Audio"),
            Patch(facecolor=c_text, edgecolor=c_text, alpha=0.55, label="Text"),
        ], loc="best")

        ax.margins(x=0.02)
        plt.tight_layout()
        plt.show()
