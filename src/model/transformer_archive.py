import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Any, Optional


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple Transformer-based classifier with an sklearn-like interface.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step (F).
    seq_len : int
        Length of each input sequence (T).
    d_model : int, default=64
        Embedding dimension for transformer.
    n_heads : int, default=4
        Number of attention heads.
    num_layers : int, default=2
        Number of TransformerEncoder layers.
    num_classes : int, default=2
        Number of output classes.
    dropout : float, default=0.1
        Dropout rate throughout the model.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    batch_size : int, default=32
        Training batch size.
    epochs : int, default=10
        Number of training epochs.
    device : str or torch.device, default='cpu'
        Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 10,
        device: Optional[str] = 'cpu'
    ) -> None:
        # Model hyper-parameters
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
        # Build network, criterion, and optimizer
        self._build_model()

    def _build_model(self) -> None:
        """
        Construct the Transformer encoder and classification head.
        """
        # Input projection: maps input_dim -> d_model
        input_proj = nn.Linear(self.input_dim, self.d_model)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu'
        )
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # Classification head: pool over time then linear to num_classes
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # input shape [B, d_model, seq_len] -> [B, d_model, 1]
            nn.Flatten(),              # [B, d_model]
            nn.Linear(self.d_model, self.num_classes)
        )
        # Combine into one module
        self.model = nn.ModuleDict({
            'input_proj': input_proj,
            'dropout': nn.Dropout(self.dropout),
            'transformer': transformer,
            'classifier': classifier
        }).to(self.device)
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TransformerClassifier':  # type: ignore
        """
        Train the TransformerClassifier.

        Parameters
        ----------
        X : ndarray of shape (N, T, F)
            Training data: N sequences, each length T with F features.
        y : ndarray of shape (N,)
            Integer class labels.

        Returns
        -------
        self
        """
        # Convert to torch tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                # Zero gradients
                self.optimizer.zero_grad()
                # Forward pass
                # 1) input_proj + dropout => [B, T, d_model]
                h = self.model['input_proj'](xb)
                h = self.model['dropout'](h)
                # 2) Transformer expects [T, B, d_model]
                h = h.permute(1, 0, 2)
                h = self.model['transformer'](h)
                # 3) Back to [B, d_model, T] for pooling
                h = h.permute(1, 2, 0)
                # 4) Classification head
                logits = self.model['classifier'](h)
                # Compute loss
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataset)
            # Optional verbose: print(f"Epoch {epoch+1}/{self.epochs}, Loss {avg_loss:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : ndarray of shape (N, T, F)
            Input sequences.

        Returns
        -------
        preds : ndarray of shape (N,)
            Predicted integer labels.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h = self.model['input_proj'](X_t)
            h = self.model['dropout'](h)
            h = h.permute(1, 0, 2)
            h = self.model['transformer'](h)
            h = h.permute(1, 2, 0)
            logits = self.model['classifier'](h)
            preds = logits.argmax(dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : ndarray of shape (N, T, F)
            Input sequences.

        Returns
        -------
        proba : ndarray of shape (N, num_classes)
            Probability of each class.
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h = self.model['input_proj'](X_t)
            h = self.model['dropout'](h)
            h = h.permute(1, 0, 2)
            h = self.model['transformer'](h)
            h = h.permute(1, 2, 0)
            logits = self.model['classifier'](h)
            proba = nn.functional.softmax(logits, dim=1)
        return proba.cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float: # type: ignore
        """
        Compute accuracy.

        Parameters
        ----------
        X : ndarray of shape (N, T, F)
        y : ndarray of shape (N,)

        Returns
        -------
        acc : float
            Classification accuracy.
        """
        preds = self.predict(X)
        return float((preds == y).mean())
