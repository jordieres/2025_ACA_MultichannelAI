import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class AnswerPlotter:
    @staticmethod
    def plot_answers_by_topic(df: pd.DataFrame):
        df = df.copy()
        df["answered"] = df["answered"].str.lower()

        counts = df.groupby(["topic_label", "answered"]).size().unstack(fill_value=0)
        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

        counts["total"] = counts.sum(axis=1)
        counts["evasive_ratio"] = (
            counts.get("no", 0) + counts.get("partially", 0)
        ) / counts["total"]

        colors = {
            "yes": "#4CAF50",
            "partially": "#FFC107",
            "no": "#F44336"
        }

        fig, ax = plt.subplots(figsize=(12, 7))

        bottom = None
        for answer_type in ["no", "partially", "yes"]:
            if answer_type not in counts.columns:
                continue

            ax.barh(
                counts.index,
                counts[answer_type],
                left=bottom,
                label=answer_type.capitalize(),
                color=colors[answer_type],
                edgecolor="white"
            )
            bottom = counts[answer_type] if bottom is None else bottom + counts[answer_type]

        # Add evasive ratio annotations
        for i, (_, row) in enumerate(counts.iterrows()):
            x_pos = row["total"] + counts["total"].max() * 0.01
            ax.text(
                x_pos,
                i,
                f"{row['evasive_ratio']:.1%}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color="black"
            )

        xmax = counts["total"].max()
        ax.set_xlim(0, xmax * 1.12)

        ax.set_xlabel("Number of questions")
        ax.set_ylabel("Topic")
        ax.set_title("Distribution of Questions by Topic and Answer Type")

        # ---- Legend fix (this is the key part) ----
        evasive_proxy = Line2D(
            [0], [0],
            linestyle="None",
            marker=r"$\%$",
            markersize=11,
            color="black",
            label="Evasive ratio (No + Partially)"
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.append(evasive_proxy)

        ax.legend(
            handles=handles,
            title="Answer Type",
            loc="best"
        )

        plt.tight_layout()
        plt.show()
