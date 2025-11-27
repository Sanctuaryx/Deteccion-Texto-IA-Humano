# iatd/models/custom_bilstm.py
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        pad_index: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_index,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # capa intermedia + ReLU
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)  # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        attn_scores = self.attn_w(out).squeeze(-1)  # (B, L)

        # máscara de padding: donde length < posición, ponemos -inf
        mask = torch.arange(out.size(1))[None, :].to(lengths.device) >= lengths[:, None]
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)

        # vector de texto = combinación lineal de hidden states
        text_vec = (out * attn_weights).sum(dim=1)  # (B, 2*H)

        text_vec = self.dropout(text_vec)
        logits = self.fc(text_vec).squeeze(1)
        return logits
