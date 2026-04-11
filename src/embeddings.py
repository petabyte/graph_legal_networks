from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.graph_builder import _strip_html

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def _mean_pool(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
        mask_expanded.sum(dim=1), min=1e-9
    )


class LegalBertEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of plain-text strings.
        Strips HTML automatically before tokenizing.
        Returns (N, 768) float32 mean-pooled embeddings.
        Uses GPU automatically when available.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [_strip_html(t) for t in texts[i: i + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            emb = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)


def cosine_similarity_pairs(
    embeddings: np.ndarray, pairs: list[tuple[int, int]]
) -> np.ndarray:
    """
    Return cosine similarity for each (i, j) index pair.
    embeddings: (N, D) — rows are already mean-pooled
    Returns: (len(pairs),) float array
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-9, None)
    return np.array([float(np.dot(normed[i], normed[j])) for i, j in pairs], dtype=float)
