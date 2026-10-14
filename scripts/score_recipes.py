#!/usr/bin/env python
"""Stage 2 of the data pipeline (GPU): LLM-as-judge filtering.

The base instruct model reads each candidate recipe and rates it 1-5. Instead of sampling
an answer we take the next-token distribution over the digits "1".."5" after the
generation prompt and use the expected value as a continuous score (one forward pass per
recipe, no generation - much cheaper and deterministic).

  python scripts/score_recipes.py --model yandex/YandexGPT-5-Lite-8B-instruct
  -> data/clean/scored.parquet (all candidates with judge_score)
  -> data/clean/train.parquet  (top --top_k with judge_score >= --min_score)

Work is saved in chunks (data/clean/scored_parts/), so a job killed by the time limit
resumes where it stopped.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from recipe_llm.chat import ChatFormatter  # noqa: E402
from recipe_llm.generation import load_model  # noqa: E402
from recipe_llm.prompts import format_recipe  # noqa: E402

JUDGE_SYSTEM = (
    "Ты — строгий шеф-повар и редактор кулинарного сайта. Тебе показывают рецепт, "
    "написанный пользователем. Оцени его по шкале от 1 до 5:\n"
    "1 — бессмыслица, ошибки, блюдо так не приготовить;\n"
    "2 — серьёзные проблемы: пропущены важные шаги или ингредиенты, странные количества;\n"
    "3 — приготовить можно, но есть заметные недочёты;\n"
    "4 — хороший понятный рецепт с мелкими недочётами;\n"
    "5 — отличный рецепт: соответствует названию, логичные шаги, разумные количества.\n"
    "Ответь одной цифрой."
)


def digit_ids(fmt: ChatFormatter) -> tuple[list[int], list[int]]:
    """Token ids of the answers "1".."5" as the model would start them after the prompt.
    Returns (shared_prefix_ids, [id_for_1, ..., id_for_5])."""
    seqs = [fmt.completion_ids(str(d))[:-1] for d in range(1, 6)]  # without EOS
    prefix = []
    for i in range(min(map(len, seqs))):
        col = {s[i] for s in seqs}
        if len(col) == 1:
            prefix.append(seqs[0][i])
        else:
            return prefix, [s[i] for s in seqs]
    raise ValueError(f"Cannot find distinct digit tokens: {seqs}")


@torch.no_grad()
def score_batch(model, fmt, recipes: list[str], prefix: list[int], digits: list[int]) -> list[tuple[float, float]]:
    batch = [fmt.prompt_ids("Рецепт:\n\n" + r) + prefix for r in recipes]
    input_ids, mask = fmt.left_pad(batch)
    dev = next(model.parameters()).device
    kw = dict(input_ids=input_ids.to(dev), attention_mask=mask.to(dev))
    try:  # only the last position's logits: saves ~2 GB for a 129k vocabulary
        logits = model(**kw, logits_to_keep=1).logits[:, -1, :].float()
    except TypeError:
        logits = model(**kw).logits[:, -1, :].float()
    probs = torch.softmax(logits, dim=-1)[:, digits]  # (B, 5)
    mass = probs.sum(-1)
    exp = (probs / mass.unsqueeze(-1) * torch.arange(1, 6, device=dev)).sum(-1)
    return list(zip(exp.tolist(), mass.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yandex/YandexGPT-5-Lite-8B-instruct")
    ap.add_argument("--input", default=os.path.join(ROOT, "data/clean/candidates.parquet"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data/clean"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--max_recipe_chars", type=int, default=3500)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N (debug)")
    ap.add_argument("--top_k", type=int, default=10000)
    ap.add_argument("--min_score", type=float, default=3.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--newline_token", default="auto")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="this job scores chunks with index %% num_shards == shard")
    ap.add_argument("--num_shards", type=int, default=1, help="run N jobs on N GPUs; the last one to finish merges")
    a = ap.parse_args()

    df = pd.read_parquet(a.input)
    if a.limit:
        df = df.head(a.limit)
    parts_dir = os.path.join(a.out_dir, "scored_parts")
    os.makedirs(parts_dir, exist_ok=True)

    model, tok, fmt = load_model(a.model, None, a.dtype, a.device, a.newline_token, JUDGE_SYSTEM)
    prefix, digits = digit_ids(fmt)
    print(f"digit tokens: prefix={tok.convert_ids_to_tokens(prefix)} digits={tok.convert_ids_to_tokens(digits)}")
    print("judge prompt example:\n" + fmt.prompt_text("Рецепт:\n\n<текст рецепта>"))

    t0, done = time.time(), 0
    n_chunks = math.ceil(len(df) / a.chunk_size)
    mine = [c for c in range(n_chunks) if c % a.num_shards == a.shard]
    print(f"shard {a.shard}/{a.num_shards}: {len(mine)} of {n_chunks} chunks")
    for c in mine:
        part = os.path.join(parts_dir, f"part_{c:05d}.parquet")
        chunk = df.iloc[c * a.chunk_size:(c + 1) * a.chunk_size]
        if os.path.exists(part):
            done += len(chunk)
            continue
        texts = [format_recipe(r, include_description=True)[:a.max_recipe_chars] for r in chunk.to_dict("records")]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths -> less padding
        scores = [None] * len(texts)
        for i in range(0, len(order), a.batch_size):
            idx = order[i:i + a.batch_size]
            for j, s in zip(idx, score_batch(model, fmt, [texts[k] for k in idx], prefix, digits)):
                scores[j] = s
        out = chunk[["recipe_id"]].copy()
        out["judge_score"] = [s[0] for s in scores]
        out["judge_mass"] = [s[1] for s in scores]
        out.to_parquet(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)  # atomic: other shards never see a half-written file
        done += len(chunk)
        rate = done / max(1e-9, time.time() - t0)
        print(f"[chunk {c + 1}/{n_chunks}] {done} recipes by this shard, {rate:.1f}/s, "
              f"mean score {out['judge_score'].mean():.2f}, digit mass {out['judge_mass'].mean():.2f}", flush=True)

    del model
    missing = [c for c in range(n_chunks) if not os.path.exists(os.path.join(parts_dir, f"part_{c:05d}.parquet"))]
    if missing:
        print(f"shard {a.shard} finished; {len(missing)} chunks still being scored by other shards - "
              "the last shard to finish will merge.")
        return
    import fcntl

    with open(os.path.join(a.out_dir, ".merge.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # two shards finishing at once merge one after another
        merge_and_select(a, df, parts_dir, n_chunks)


def merge_and_select(a, df, parts_dir, n_chunks):
    parts = [os.path.join(parts_dir, f"part_{c:05d}.parquet") for c in range(n_chunks)]
    scores = pd.concat([pd.read_parquet(p) for p in parts])
    scored = df.merge(scores.drop_duplicates("recipe_id"), on="recipe_id", how="inner")
    scored.to_parquet(os.path.join(a.out_dir, "scored.parquet"), index=False)

    train = scored[scored["judge_score"] >= a.min_score].sort_values("judge_score", ascending=False).head(a.top_k)
    train_path = os.path.join(a.out_dir, "train.parquet")
    train.to_parquet(train_path, index=False)

    hist = scored["judge_score"].round().value_counts().sort_index().to_dict()
    stats = {
        "scored": len(scored), "selected": len(train),
        "score_mean": round(float(scored["judge_score"].mean()), 3),
        "score_selected_min": round(float(train["judge_score"].min()), 3) if len(train) else None,
        "digit_mass_mean": round(float(scored["judge_mass"].mean()), 3),
        "histogram": {str(int(k)): int(v) for k, v in hist.items()},
    }
    with open(os.path.join(a.out_dir, "judge_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["digit_mass_mean"] < 0.5:
        print("[warn] low probability mass on digits - the judge prompt/answer format may not fit this model")

    # Examples for the report: what the judge likes and rejects.
    def examples(frame, title):
        rows = frame.to_dict("records")
        body = "\n\n---\n\n".join(f"**score {r['judge_score']:.2f}**\n\n{format_recipe(r)}" for r in rows)
        return f"# {title}\n\n{body}\n"

    ordered = scored.sort_values("judge_score")
    with open(os.path.join(a.out_dir, "judge_examples.md"), "w", encoding="utf-8") as f:
        f.write(examples(ordered.tail(5).iloc[::-1], "Highest judge scores") + "\n\n"
                + examples(ordered.head(5), "Lowest judge scores"))
    print(f"-> {train_path} ({len(train)} recipes)")

    if a.mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("recipe-data")
        with mlflow.start_run(run_name="llm_judge"):
            mlflow.log_params({"judge_model": a.model, "top_k": a.top_k, "min_score": a.min_score,
                               "max_recipe_chars": a.max_recipe_chars})
            mlflow.log_metrics({k: v for k, v in stats.items() if isinstance(v, (int, float))})
            mlflow.log_dict(stats, "judge_stats.json")
            mlflow.log_text(JUDGE_SYSTEM, "judge_prompt.txt")
            mlflow.log_artifact(os.path.join(a.out_dir, "judge_examples.md"))


if __name__ == "__main__":
    main()
