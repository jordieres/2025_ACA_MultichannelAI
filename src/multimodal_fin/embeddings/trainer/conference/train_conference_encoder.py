import torch
import logging
import optuna
from torch.utils.data import DataLoader

from multimodal_fin.embeddings.builder.node_encoder import NodeEncoder
from multimodal_fin.embeddings.builder.conference_encoder import ConferenceEncoder
from multimodal_fin.embeddings.builder.feature_extractor import FeatureExtractor
from multimodal_fin.training.Conference.ConferenceContrastiveDataset import ConferenceContrastiveDataset
from multimodal_fin.training.nt_xent_loss import nt_xent_loss

logger = logging.getLogger(__name__)

class ConferenceEncoderTrainer:
    """
    Trainer class for the ConferenceEncoder model using contrastive learning.
    
    This trainer optimizes the conference-level encoder using node embeddings
    from a pre-trained SentenceAttentionEncoder, and performs hyperparameter
    tuning via Optuna.
    """

    def __init__(
        self,
        json_paths,
        sentence_encoder_path,
        node_hidden_dim=128,
        node_meta_dim=32,
        node_d_output=512,
        save_path="conference_encoder_best.pt",
        device=None,
        batch_size=4,
        optuna_epochs=10,
        final_epochs=50,
    ):
        """
        Initializes the trainer with dataset paths and model parameters.

        Args:
            json_paths (List[str]): List of paths to JSON files.
            sentence_encoder_path (str): Path to pretrained sentence encoder weights.
            node_hidden_dim (int): Hidden dimension for NodeEncoder.
            node_meta_dim (int): Dimension for metadata projection.
            node_d_output (int): Final output dimension of NodeEncoder.
            save_path (str): File path to save the best ConferenceEncoder.
            device (str, optional): Device to use ("cuda" or "cpu").
            batch_size (int): Batch size for training.
            optuna_epochs (int): Number of epochs during Optuna optimization.
            final_epochs (int): Final number of training epochs.
        """
        self.json_paths = json_paths
        self.sentence_encoder_path = sentence_encoder_path
        self.node_hidden_dim = node_hidden_dim
        self.node_meta_dim = node_meta_dim
        self.node_d_output = node_d_output
        self.save_path = save_path

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.optuna_epochs = optuna_epochs
        self.final_epochs = final_epochs

        self.best_params = None

    def _build_components(self, trial_params=None):
        """Constructs dataset, model and dataloader based on parameters."""
        params = trial_params or self.best_params
        assert params, "Hyperparameters must be provided."

        node_encoder = NodeEncoder(
            device=self.device,
            hidden_dim=self.node_hidden_dim,
            meta_dim=self.node_meta_dim,
            d_output=self.node_d_output,
            weights_path=self.sentence_encoder_path
        ).to(self.device)

        conference_encoder = ConferenceEncoder(
            input_dim=self.node_d_output,
            hidden_dim=params["hidden_dim_conf"],
            n_heads=params["n_heads"],
            d_output=self.node_d_output
        ).to(self.device)

        extractor = FeatureExtractor(
            categories_10k=node_encoder.categories_10k,
            qa_categories=node_encoder.qa_categories,
            max_num_coherences=node_encoder.max_num_coherences
        )

        dataset = ConferenceContrastiveDataset(
            json_paths=self.json_paths,
            extractor=extractor,
            node_encoder=node_encoder,
            conference_encoder=conference_encoder,
            device=self.device
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return node_encoder, conference_encoder, dataloader

    def _objective(self, trial):
        """
        Objective function for Optuna hyperparameter search.

        Args:
            trial (optuna.trial.Trial): Current trial object.

        Returns:
            float: Final loss for the trial.
        """
        trial_params = {
            "hidden_dim_conf": trial.suggest_categorical("hidden_dim_conf", [128, 256, 512, 1024]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        }

        _, conference_encoder, dataloader = self._build_components(trial_params)
        optimizer = torch.optim.Adam(conference_encoder.parameters(), lr=trial_params["lr"])

        for epoch in range(self.optuna_epochs):
            conference_encoder.train()
            total_loss = 0.0

            for emb1, emb2 in dataloader:
                emb1, emb2 = emb1.to(self.device), emb2.to(self.device)
                out1 = conference_encoder(emb1)
                out2 = conference_encoder(emb2)
                loss = nt_xent_loss(out1, out2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            trial.report(avg_loss, step=epoch)
            logger.debug(f"[Trial {trial.number}] Epoch {epoch+1} - Loss: {avg_loss:.4f}")

            if trial.should_prune():
                logger.info(f"⏹️ Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

        return avg_loss

    def optimize(self, n_trials=30, direction="minimize"):
        """
        Runs hyperparameter optimization using Optuna.

        Args:
            n_trials (int): Number of trials.
            direction (str): Optimization direction ("minimize" or "maximize").

        Returns:
            dict: Best hyperparameters found.
        """
        logger.info("🔍 Starting Optuna hyperparameter optimization...")
        study = optuna.create_study(direction=direction)
        study.optimize(self._objective, n_trials=n_trials)

        self.best_params = study.best_params
        logger.info(f"🏆 Best hyperparameters: {self.best_params}")
        return self.best_params

    def train(self):
        """
        Trains the ConferenceEncoder using the best hyperparameters.

        Returns:
            torch.nn.Module: Trained ConferenceEncoder.
        """
        if not self.best_params:
            raise RuntimeError("Call optimize() before train().")

        _, conference_encoder, dataloader = self._build_components()
        optimizer = torch.optim.Adam(conference_encoder.parameters(), lr=self.best_params["lr"])

        logger.info("🚀 Starting final training...")
        for epoch in range(self.final_epochs):
            conference_encoder.train()
            total_loss = 0.0

            for emb1, emb2 in dataloader:
                emb1, emb2 = emb1.to(self.device), emb2.to(self.device)
                out1 = conference_encoder(emb1)
                out2 = conference_encoder(emb2)
                loss = nt_xent_loss(out1, out2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(dataloader)
            logger.info(f"📦 Epoch {epoch+1}/{self.final_epochs} - Loss: {avg:.4f}")

        torch.save(conference_encoder.state_dict(), self.save_path)
        logger.info(f"✅ ConferenceEncoder weights saved to {self.save_path}")
        return conference_encoder
