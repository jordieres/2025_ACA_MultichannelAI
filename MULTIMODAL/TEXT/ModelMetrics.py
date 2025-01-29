from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    """
    A class for evaluating and visualizing classification performance metrics for a model.

    Attributes:
        model_name (str): The name of the model being evaluated.
        results_path (str): The directory where results, including plots, will be saved.
    """
    
    model_name: str
    results_path: str

    def __post_init__(self):
        """
        Initializes the class by creating the results directory if it does not exist.
        """
        os.makedirs(self.results_path, exist_ok=True)

    def get_aditional_data(self, df) -> dict:
        """
        Computes additional classification data, including a classification report and a confusion matrix.

        Args:
            df (pd.DataFrame): A DataFrame containing the true labels ('label') and predicted classifications ('classification').

        Returns:
            dict: A dictionary containing:
                - labels (list): Sorted unique labels in the dataset.
                - report (dict): Classification report with precision, recall, and F1-score.
                - conf_matrix (np.array): Confusion matrix.
        """
        labels = sorted(df['label'].unique())
        report = classification_report(df['label'], df['classification'], target_names=labels, output_dict=True)
        conf_matrix = confusion_matrix(df['label'], df['classification'])
        return {'labels': labels, 'report': report, 'conf_matrix': conf_matrix}
    
    def get_metrics(self, df) -> dict:
        """
        Computes global classification metrics including accuracy, precision, recall, and F1-score.

        Args:
            df (pd.DataFrame): A DataFrame containing the true labels ('label') and predicted classifications ('classification').

        Returns:
            dict: A dictionary containing:
                - accuracy (float): Overall classification accuracy.
                - precision (float): Macro-averaged precision score.
                - recall (float): Macro-averaged recall score.
                - f1_score (float): Macro-averaged F1 score.
        """
        accuracy = accuracy_score(df['label'], df['classification'])
        precision_global = precision_score(df['label'], df['classification'], average='macro', zero_division=0.0)
        recall_global = recall_score(df['label'], df['classification'], average='macro', zero_division=0.0)
        f1_score_global = f1_score(df['label'], df['classification'], average='macro', zero_division=0.0)
        return {
            'accuracy': accuracy,
            'precision': precision_global,
            'recall': recall_global,
            'f1_score': f1_score_global
        }
    
    def plots_metrics(self, metrics, aditional_data):
        """
        Generates and saves plots for classification metrics:
        1. A confusion matrix heatmap.
        2. A bar chart displaying global classification metrics.

        Args:
            metrics (dict): Dictionary of computed classification metrics.
            aditional_data (dict): Dictionary containing additional data, including labels and the confusion matrix.

        Saves:
            - A PNG file with the visualized results in the specified results directory.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Confusion Matrix
        sns.heatmap(
            aditional_data['conf_matrix'], annot=True, fmt="d", cmap="Blues",
            xticklabels=aditional_data['labels'], yticklabels=aditional_data['labels'], cbar=False, ax=axes[0]
        )
        axes[0].set_title(f"Confusion Matrix | {self.model_name}", fontsize=14)
        axes[0].set_xlabel("Predicted", fontsize=12)
        axes[0].set_ylabel("True", fontsize=12)

        # Global Metrics
        bars = axes[1].bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha='center', fontsize=10)
        axes[1].set_title(f"Global Metrics | {self.model_name}", fontsize=14)
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        fig.savefig(self.results_path + f'{self.model_name}.png', bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def plot_f1_vs_time_all_models(self, df):
        """
        Generates a scatter plot showing the relationship between elapsed time and F1-score for multiple models.

        Args:
            df (pd.DataFrame): A DataFrame containing:
                - 'elapsed_time' (float): The time taken to run the model.
                - 'f1_score' (float): The F1-score of the model.
                - 'model_name' (str): The name of the model.

        Displays:
            - A scatter plot with model names labeled for interpretation.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="elapsed_time", y="f1_score", color="blue", s=100, alpha=0.7)

        # Add labels to each point with the model name, oriented diagonally
        for _, row in df.iterrows():
            plt.text(row["elapsed_time"], row["f1_score"], row["model_name"], 
                     fontsize=9, ha="left", va="center", rotation=30)

        plt.xlabel("Elapsed Time")
        plt.ylabel("F1 Score")
        plt.title("Relationship between Elapsed Time and F1 Score")
        plt.grid(True)

        plt.show()

    def get_results(self, df):
        """
        Computes classification metrics, generates plots, and returns evaluation results.

        Args:
            df (pd.DataFrame): A DataFrame containing the true labels ('label') and predicted classifications ('classification').

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1-score.
        
        Additionally:
            - Generates and saves a confusion matrix heatmap.
            - Generates and saves a bar plot of classification metrics.
        """
        metrics = self.get_metrics(df)
        aditional_data = self.get_aditional_data(df)
        self.plots_metrics(metrics, aditional_data)
        return metrics
