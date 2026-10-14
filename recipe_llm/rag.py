"""Retrieval-augmented generation: find similar real recipes and put them into the prompt.

Index: every cleaned povarenok recipe is embedded once (title + ingredients + beginning of the steps)
with a multilingual sentence encoder (default intfloat/multilingual-e5-large). Search is exact
cosine similarity over all vectors (~10^5 x 1024 in fp16 on the GPU - milliseconds, no FAISS needed),
then a small popularity bonus breaks ties between near-identical matches.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .prompts import clean_text, format_recipe, get_ingredients, get_steps

DEFAULT_ENCODER = "intfloat/multilingual-e5-large"

RAG_INSTRUCTION = (
    "Ниже — похожие рецепты из кулинарной базы. Используй их как опору: возьми правильные ингредиенты, "
    "количества и последовательность действий, исправь ошибки и не копируй дословно. "
    "Если рецепты не подходят к запросу, опирайся на свои знания."
)


def passage_text(rec: dict, max_chars: int = 600) -> str:
    """What gets embedded for a recipe."""
    ings = ", ".join(clean_text(i.get("name")) for i in get_ingredients(rec) if clean_text(i.get("name")))
    steps = " ".join(get_steps(rec))[:max_chars]
    return f"{clean_text(rec['title'])}. Ингредиенты: {ings}. {steps}"


class Encoder:
    """Sentence embeddings with plain transformers (mean pooling for E5, CLS for BGE)."""

    def __init__(self, name: str = DEFAULT_ENCODER, device: Optional[str] = None, max_length: int = 512):
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        low = name.lower()
        self.e5 = "e5" in low
        self.cls_pooling = "bge" in low

    @torch.no_grad()
    def encode(self, texts: list[str], kind: str = "passage", batch_size: int = 64) -> np.ndarray:
        prefix = f"{kind}: " if self.e5 else ""
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tok([prefix + texts[j] for j in idx], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state.float()
            if self.cls_pooling:
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append((idx, torch.nn.functional.normalize(v, dim=-1).cpu().numpy()))
        res = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, v in out:
            res[idx] = v
        return res


def build_index(records: list[dict], out_dir: str, encoder: Encoder, batch_size: int = 64) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    texts = [passage_text(r) for r in records]
    emb = encoder.encode(texts, "passage", batch_size)
    np.save(os.path.join(out_dir, "embeddings.npy"), emb.astype(np.float16))
    meta = pd.DataFrame({
        "recipe_id": [r.get("recipe_id") for r in records],
        "title": [clean_text(r["title"]) for r in records],
        "views": [float(r.get("views") or 0) for r in records],
        "recipe_md": [format_recipe(r, include_description=False) for r in records],
    })
    meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    info = {"encoder": encoder.name, "n": len(records), "dim": int(emb.shape[1])}
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


class Retriever:
    def __init__(self, index_dir: str, device: Optional[str] = None, encoder: Optional[Encoder] = None,
                 pop_weight: float = 0.03):
        with open(os.path.join(index_dir, "info.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.encoder = encoder or Encoder(self.info["encoder"], device)
        dev = self.encoder.device
        emb = np.load(os.path.join(index_dir, "embeddings.npy"))
        self.emb = torch.from_numpy(emb).to(dev, dtype=torch.float16 if dev.startswith("cuda") else torch.float32)
        self.meta = pd.read_parquet(os.path.join(index_dir, "meta.parquet"))
        lv = np.log1p(self.meta["views"].to_numpy(dtype=np.float64))
        self.pop = torch.tensor(lv / max(lv.max(), 1e-9), dtype=torch.float32, device=dev)
        self.pop_weight = pop_weight

    @torch.no_grad()
    def search(self, query: str, k: int = 3, candidates: int = 50) -> list[dict]:
        q = torch.from_numpy(self.encoder.encode([query], "query")).to(self.emb.device, self.emb.dtype)
        sims = (self.emb @ q[0]).float()
        top = torch.topk(sims, min(candidates, len(sims)))
        scored = sorted(((float(s) + self.pop_weight * float(self.pop[i]), int(i), float(s))
                         for s, i in zip(top.values, top.indices)), reverse=True)
        seen, out = set(), []
        for score, i, sim in scored:
            title = self.meta.at[i, "title"].lower()
            if title in seen:
                continue
            seen.add(title)
            rid = self.meta.at[i, "recipe_id"]
            out.append({"recipe_id": None if pd.isna(rid) else int(rid), "title": self.meta.at[i, "title"],
                        "similarity": round(sim, 4), "recipe_md": self.meta.at[i, "recipe_md"]})
            if len(out) == k:
                break
        return out

    def augment(self, user_text: str, k: int = 3, max_chars_per_recipe: int = 1500) -> tuple[str, list[dict]]:
        """Returns (prompt with retrieved recipes, retrieved list)."""
        hits = self.search(user_text, k)
        if not hits:
            return user_text, hits
        refs = "\n\n".join(f"Рецепт {n}:\n{h['recipe_md'][:max_chars_per_recipe]}" for n, h in enumerate(hits, 1))
        return f"{RAG_INSTRUCTION}\n\n{refs}\n\nЗапрос: {user_text}", hits


def load_retriever(index_dir: Optional[str], device: Optional[str] = None) -> Optional[Retriever]:
    if not index_dir:
        return None
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return Retriever(index_dir, device)
