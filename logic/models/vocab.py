from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizador simple y robusto para castellano:
    - separa palabras y signos de puntuación
    - mantiene acentos/ñ
    """
    if not text:
        return []
    text = text.strip().lower()
    return _TOKEN_RE.findall(text)


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_index: int
    unk_index: int

    @staticmethod
    def build(
        texts: Sequence[str],
        min_freq: int = 2,
        max_size: Optional[int] = 50000,
        specials: Optional[List[str]] = None,
    ) -> "Vocab":
        """
        Construye vocabulario a partir de textos.
        """
        if specials is None:
            specials = ["<pad>", "<unk>"]

        freq: Dict[str, int] = {}
        for t in texts:
            for tok in simple_tokenize(str(t)):
                freq[tok] = freq.get(tok, 0) + 1

        # Orden: más frecuente primero, empate por orden alfabético
        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

        itos: List[str] = []
        for sp in specials:
            if sp not in itos:
                itos.append(sp)

        for tok, c in items:
            if c < min_freq:
                break
            if tok in itos:
                continue
            itos.append(tok)
            if max_size is not None and len(itos) >= max_size:
                break

        stoi = {tok: i for i, tok in enumerate(itos)}

        pad_index = stoi.get("<pad>", 0)
        unk_index = stoi.get("<unk>", 1 if len(itos) > 1 else 0)

        return Vocab(stoi=stoi, itos=itos, pad_index=pad_index, unk_index=unk_index)

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(tok, self.unk_index) for tok in tokens]

    def encode_text(self, text: str) -> List[int]:
        return self.encode_tokens(simple_tokenize(text))

    def decode_ids(self, ids: Sequence[int]) -> List[str]:
        out = []
        for i in ids:
            if 0 <= int(i) < len(self.itos):
                out.append(self.itos[int(i)])
            else:
                out.append(self.itos[self.unk_index])
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "itos": self.itos,
            "pad_index": int(self.pad_index),
            "unk_index": int(self.unk_index),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Vocab":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        itos = list(payload["itos"])
        stoi = {tok: i for i, tok in enumerate(itos)}
        return Vocab(
            stoi=stoi,
            itos=itos,
            pad_index=int(payload["pad_index"]),
            unk_index=int(payload["unk_index"]),
        )
