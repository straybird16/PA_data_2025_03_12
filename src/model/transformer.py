import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any, List, Optional, Tuple


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


class TransformerModel(nn.Module):
    """
    Core Transformer model with encoder and optional decoder for reconstruction.
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            #activation='gelu',
            batch_first=True  # input is [B, T, F]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Decoder transformer for reconstruction tasks (maps model dim back to input feature space)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                #activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        # Pooling + classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # pool over time dimension
            nn.Flatten(),              # flatten to [B, d_model]
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.num_classes),      # single logit; wrapper will handle multi-class
        )
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        task: str = 'classify',
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass supports multiple tasks:
         - 'classify': output class logits
         - 'reconstruct': output full-sequence reconstruction
         - 'masked_reconstruct': reconstruct only masked positions
         - 'complete': predict second half from entire input
        """
        # x: [B, T, F]
        B, T, F = x.size()
        # Optional masking: set masked positions to zero
        input = x.clone()
        if task == 'masked_reconstruct' and mask is not None:
            x = x.clone()
            x[mask] = 0.0

        # Input projection + dropout
        h = self.input_proj(x)               # [B, T, d_model]
        h = self.dropout(h)
        # Transformer expects [B, T, d_model]
        h = self.transformer(h)  # [B, T, d_model]

        if task == 'classify':
            # Pool and classify
            h_t = h.permute(0, 2, 1)           # [B, d_model, T]
            logits = self.classifier(h_t)      # [B, 1]
            return logits.squeeze(1)           # [B]

        # For reconstruction tasks, decode sequence
        recon = self.decoder(input, h)               # [B, T, input_dim]

        if task == 'reconstruct':
            return recon                      # full reconstruction
        if task == 'masked_reconstruct':
            # Only compute loss on masked positions
            return recon
        if task == 'complete':
            # Return predictions for second half
            half = T // 2
            return recon[:, half:, :]        # [B, T-half, input_dim]

        raise ValueError(f"Unknown task: {task}")


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper for TransformerModel with sklearn-like API.

    Supports supervised training and unsupervised pretraining on reconstruction tasks.
    """
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        device: Optional[str] = 'cuda:0' if torch.cuda.is_available() else 'cpu',
        seed: int = 42
    ) -> None:
        set_seed(seed)
        # Save hyperparameters
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        # Set device
        self.device = torch.device(device if isinstance(device, str) else str(device))
        # Build model
        self.model = TransformerModel(
            input_dim, seq_len, d_model, n_heads, num_layers, dropout, num_classes=num_classes
        ).to(self.device)
        # Optimizer and criterions
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.classification_loss = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()

    def pretrain(
        self,
        X: np.ndarray,
        tasks: List[str] = ['reconstruct', 'masked_reconstruct', 'complete'],
        mask_prob: float = 0.15
    ) -> 'TransformerClassifier':
        """
        Unsupervised pretraining on reconstruction-based tasks.

        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (N, T, F) for unsupervised pretraining.
        tasks : list of str
            Tasks to perform: 'reconstruct', 'masked_reconstruct', 'complete'.
        mask_prob : float
            Probability of masking each token for masked reconstruction.

        Returns:
        --------
        self
        """
        # Prepare data loader
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        pretrain_epochs = 5
        for epoch in range(pretrain_epochs):
            total_loss = 0.0
            for (xb,) in loader:
                self.optimizer.zero_grad()
                batch_loss = 0.0
                # For each task, compute loss
                for task in tasks:
                    if task == 'masked_reconstruct':
                        # sample mask indices
                        mask = torch.rand(xb.shape[:2], device=self.device) < mask_prob
                        recon = self.model(xb, task=task, mask=mask)
                        # compute loss only on masked positions
                        target = xb
                        loss: torch.Tensor = self.recon_loss(recon[mask], target[mask])
                    elif task == 'complete':
                        recon = self.model(xb, task=task)
                        half = xb.size(1) // 2
                        target = xb[:, half:, :]
                        loss: torch.Tensor = self.recon_loss(recon, target)
                    elif task == 'reconstruct':
                        recon = self.model(xb, task=task)
                        loss: torch.Tensor = self.recon_loss(recon, xb)
                    else:
                        continue
                    batch_loss += loss
                batch_loss.backward() # type: ignore
                self.optimizer.step()
                total_loss += batch_loss.item() * xb.size(0) # type: ignore
            avg_loss = total_loss / len(dataset)
            # Detailed logging
            print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}, Avg Loss: {avg_loss:.4f}")
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, pretrain: bool = True) -> 'TransformerClassifier':
        """
        Supervised training on labeled data.

        Parameters:
        -----------
        X : np.ndarray of shape (N, T, F)
            Training sequences.
        y : np.ndarray of shape (N,)
            Integer labels from 0 to num_classes-1.

        Returns:
        --------
        self
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        if pretrain:
            # Pretrain on reconstruction tasks before supervised training
            self.pretrain(X, tasks=['complete', 'reconstruct', 'masked_reconstruct'])

        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb, task='classify')
                loss = self.classification_loss(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Train Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input sequences.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t, task='classify')  # [B]
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using softmax over logits.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_t, task='classify')  # [B]
            # Expand logits to shape [B, num_classes]
            logits_exp = logits.unsqueeze(1).repeat(1, self.num_classes)
            proba = nn.functional.softmax(logits_exp, dim=1)
        return proba.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float: # type: ignore
        """
        Compute classification accuracy.
        """
        preds = self.predict(X)
        return float((preds == y).mean())
