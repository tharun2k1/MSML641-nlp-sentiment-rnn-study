import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Return a torch activation module by name.
    Used for the fully-connected head (not the recurrent cell).
    """
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class SequenceClassifier(nn.Module):
    """
    Generic sequence classifier with:
      - Embedding
      - Recurrent encoder (RNN or LSTM; uni- or bi-directional)
      - FC head with configurable activation
    Outputs a single logit for BCEWithLogitsLoss.
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 64,
        n_layers: int = 2,
        rnn_type: str = "lstm",          # "rnn" or "lstm"
        bidirectional: bool = False,     # True for BiLSTM
        dropout: float = 0.5,
        fc_activation: str = "relu",
        pad_idx: int = 0
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        if rnn_type.lower() == "rnn":
            # Nonlinearity of vanilla RNN is tanh (fixed inside encoder);
            # the comparative activation is applied in the FC head.
            self.encoder = nn.RNN(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                nonlinearity="tanh",
                bidirectional=bidirectional
            )
        elif rnn_type.lower() == "lstm":
            self.encoder = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError("rnn_type must be 'rnn' or 'lstm'")

        d = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * d, hidden_size),
            get_activation(fc_activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) ,
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor of shape (B, T) with token IDs.
        Returns:
            logits: FloatTensor of shape (B, 1)
        """
        emb = self.emb(x)  # (B, T, E)

        if isinstance(self.encoder, nn.LSTM):
            out, _ = self.encoder(emb)  # out: (B, T, H*d)
        else:
            out, _ = self.encoder(emb)  # out: (B, T, H*d)

        # Last time-step features
        last = out[:, -1, :]           # (B, H*d)
        logits = self.fc(last)         # (B, 1)
        return logits


def make_model(
    architecture: str,
    vocab_size: int,
    emb_dim: int = 100,
    hidden_size: int = 64,
    n_layers: int = 2,
    dropout: float = 0.5,
    activation: str = "relu"
) -> nn.Module:
    """
    Factory for required variants:
      - "rnn"     -> vanilla RNN
      - "lstm"    -> LSTM
      - "bilstm"  -> bidirectional LSTM
    """
    arch = architecture.lower()
    if arch == "rnn":
        return SequenceClassifier(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            rnn_type="rnn",
            bidirectional=False,
            dropout=dropout,
            fc_activation=activation,
        )
    if arch == "lstm":
        return SequenceClassifier(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            rnn_type="lstm",
            bidirectional=False,
            dropout=dropout,
            fc_activation=activation,
        )
    if arch == "bilstm":
        return SequenceClassifier(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            rnn_type="lstm",
            bidirectional=True,
            dropout=dropout,
            fc_activation=activation,
        )
    raise ValueError(f"Unknown architecture: {architecture}")
