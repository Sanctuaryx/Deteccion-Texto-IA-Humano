from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM + atención + FC binario.
    - Salida: logits (antes de sigmoid)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_index: int = 0,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_index,
        )

        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"pretrained_embeddings shape={tuple(pretrained_embeddings.shape)} "
                    f"pero se esperaba ({vocab_size}, {embed_dim})"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_w = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L)
        lengths: (B,)
        returns logits: (B,)
        """
        emb = self.embedding(input_ids)  # (B, L, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: (B, L, 2H)

        # Atención (enmascarando padding)
        attn_scores = self.attn_w(out).squeeze(-1)  # (B, L)

        # mask: True en posiciones padding
        L = out.size(1)
        mask = torch.arange(L, device=lengths.device)[None, :] >= lengths[:, None]
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        text_vec = (out * attn_weights).sum(dim=1)  # (B, 2H)

        text_vec = self.dropout(text_vec)
        logits = self.fc(text_vec).squeeze(1)  # (B,)
        return logits
