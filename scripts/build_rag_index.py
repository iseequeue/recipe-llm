#!/usr/bin/env python
"""Build the RAG index over all cleaned povarenok recipes (GPU recommended, ~10-20 min on a V100).

  python scripts/build_rag_index.py                      # -> data/rag_index/{embeddings.npy,meta.parquet,info.json}
  python scripts/build_rag_index.py --encoder BAAI/bge-m3

Uses the same rule-based filters as scripts/build_dataset.py (but no popularity cut and no LLM judge:
for retrieval, coverage matters more than polish), deduplicated by title.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from recipe_llm.rag import DEFAULT_ENCODER, Encoder, build_index  # noqa: E402
from recipe_llm.textutil import normalize_title  # noqa: E402


def load_build_dataset():
    spec = importlib.util.spec_from_file_location("build_dataset", os.path.join(ROOT, "scripts", "build_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", default=os.path.join(ROOT, "data/povarenok/data/*.parquet"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data/rag_index"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="index only N recipes (debug)")
    ap.add_argument("--filter_args", default="",
                    help='build_dataset.py filter overrides as one string, e.g. --filter_args="--min_steps 4"')
    a = ap.parse_args()

    bd = load_build_dataset()
    fa = bd.make_parser().parse_args(shlex.split(a.filter_args))
    df = bd.read_parquets(a.recipes)
    df["recipe_id"] = df["page_url"].map(bd.recipe_id)
    recs = [r for r in df.to_dict("records") if bd.reject_reason(r, fa) is None]
    n_filtered = len(recs)
    best = {}
    for r in recs:  # dedupe by title, keep the most viewed version
        key = normalize_title(r["title"])
        if key not in best or (r.get("views") or 0) > (best[key].get("views") or 0):
            best[key] = r
    recs = list(best.values())
    if a.limit:
        recs = recs[:a.limit]
    print(f"raw {len(df)} -> filtered {n_filtered} -> deduped {len(recs)} recipes to index", flush=True)

    t0 = time.time()
    enc = Encoder(a.encoder, a.device)
    info = build_index(recs, a.out_dir, enc, a.batch_size)
    print(f"index: {info} in {time.time() - t0:.0f}s -> {a.out_dir}")

    from recipe_llm.rag import Retriever

    r = Retriever(a.out_dir, encoder=enc)
    for q in ["Напиши рецепт: Борщ", "У меня есть: курица, рис и морковь. Что можно приготовить?"]:
        print(f"\n> {q}")
        for h in r.search(q, 3):
            print(f"   {h['similarity']:.3f}  {h['title']}")


if __name__ == "__main__":
    main()
