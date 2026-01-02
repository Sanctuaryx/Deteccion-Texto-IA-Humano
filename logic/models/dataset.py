from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .vocab import Vocab, simple_tokenize


@dataclass
class TextExample:
    text: str
    label: int  # 0 humano, 1 IA


class TextDataset(Dataset):
    """
    Dataset que:
    - tokeniza
    - convierte a ids con el vocab
    - trunca a max_len
    """

    def __init__(self, examples: Sequence[TextExample], vocab: Vocab, max_len: int = 256) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        tokens = simple_tokenize(ex.text)
        ids = self.vocab.encode_tokens(tokens)

        if self.max_len is not None and len(ids) > self.max_len:
            ids = ids[: self.max_len]

        length = len(ids)
        if length == 0:
            # Evita secuencias vacías: usa <unk>
            ids = [self.vocab.unk_index]
            length = 1

        input_ids = torch.tensor(ids, dtype=torch.long)
        length_t = torch.tensor(length, dtype=torch.long)
        label_t = torch.tensor(float(ex.label), dtype=torch.float32)

        return input_ids, length_t, label_t


def collate_batch(batch, pad_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    batch: lista de (input_ids, length, label)
    Devuelve:
      - input_ids: (B, Lmax)
      - lengths: (B,)
      - labels: (B,)
    """
    input_ids_list, lengths_list, labels_list = zip(*batch)

    lengths = torch.stack(lengths_list)  # (B,)
    labels = torch.stack(labels_list)    # (B,)

    max_len = int(lengths.max().item())
    B = len(input_ids_list)

    padded = torch.full((B, max_len), fill_value=int(pad_index), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        padded[i, : ids.size(0)] = ids

    return padded, lengths, labels
