import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna

from typing import List, Optional, Dict
from multimodal_fin.embeddings.builder.sentence_attention_encoder import SentenceAttentionEncoder
from multimodal_fin.embeddings.trainer.node.node_contrastive_dataset import NodeContrastiveDataset
from multimodal_fin.training.nt_xent_loss import nt_xent_loss


class NodeEncoderTrainer:
    """
    Trainer for the SentenceAttentionEncoder using contrastive learning (NT-Xent loss)
    and Optuna for hyperparameter optimization.
    """

    def __init__(
        self,
        json_paths: List[str],
        input_dim: int,
        save_path: str = "best_encoder.pt",
        device: Optional[str] = None,
        batch_size: int = 16,
        optuna_epochs: int = 5,
        final_epochs: int = 100,
        seed: int = 42,
    ):
        self.json_paths = json_paths
        self.input_dim = input_dim
        self.save_path = save_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.optuna_epochs = optuna_epochs
        self.final_epochs = final_epochs
        self.seed = seed

        self.best_params: Optional[Dict] = None

        # Set seed for reproducibility
        torch.manual_seed(self.seed)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for Optuna hyperparameter search.

        Args:
            trial (optuna.trial.Trial): The trial object.

        Returns:
            float: The average loss after training for optuna_epochs.
        """

        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.1, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        dataset = NodeContrastiveDataset(self.json_paths)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        model = SentenceAttentionEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(self.optuna_epochs):
            model.train()
            total_loss = 0.0

            for view1, view2 in loader:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                z1 = model(view1)
                z2 = model(view2)

                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            trial.report(avg_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_loss

    def optimize(self, n_trials: int = 30, direction: str = "minimize") -> Dict:
        """
        Run Optuna hyperparameter search.

        Args:
            n_trials (int): Number of trials.
            direction (str): Optimization direction.

        Returns:
            Dict: Best hyperparameters found.
        """
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_params
        print(f"🏆 Best hyperparameters: {self.best_params}")
        return self.best_params

    def train(self) -> SentenceAttentionEncoder:
        """
        Train the final model using best hyperparameters found with optimize().

        Returns:
            SentenceAttentionEncoder: The trained model.
        """
        if not self.best_params:
            raise RuntimeError("Call optimize() before train().")

        dataset = NodeContrastiveDataset(self.json_paths)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        model = SentenceAttentionEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.best_params["hidden_dim"],
            n_heads=self.best_params["n_heads"],
            dropout=self.best_params["dropout"]
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.best_params["lr"])

        for epoch in range(1, self.final_epochs + 1):
            model.train()
            total_loss = 0.0

            for view1, view2 in loader:
                view1, view2 = view1.to(self.device), view2.to(self.device)
                z1 = model(view1)
                z2 = model(view2)

                loss = nt_xent_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"📈 Epoch {epoch:3d}/{self.final_epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), self.save_path)
        print(f"✅ Final model saved to: {self.save_path}")

        return model
