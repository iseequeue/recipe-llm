#!/usr/bin/env python
"""Stage 1 of the data pipeline (CPU): clean povarenok.ru recipes and pick candidates for the LLM judge.

  python scripts/build_dataset.py                       # -> data/clean/candidates.parquet + build_stats.json

Steps:
  1. join likes/ups from the second povarenok dump (dim/povarenok) by recipe id;
  2. rule-based filters (each rejection reason is counted and logged);
  3. dedupe by normalized title, keeping the most popular version;
  4. rank by popularity and keep the top --max_candidates for scripts/score_recipes.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from recipe_llm.prompts import clean_text, get_ingredients, get_steps  # noqa: E402
from recipe_llm.textutil import cyrillic_ratio, is_staple, mentions, normalize_title  # noqa: E402

ID_RE = re.compile(r"/recipes/show/(\d+)")
KEEP = ["recipe_id", "title", "description", "views", "likes", "ups", "kroshki", "ingredients",
        "cooking_time", "portions_count", "cooking_steps", "purposes", "tastes", "page_url"]


def recipe_id(url) -> int | None:
    m = ID_RE.search(str(url or ""))
    return int(m.group(1)) if m else None


def read_parquets(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"No files match {pattern}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def reject_reason(rec: dict, a) -> str | None:
    title = clean_text(rec.get("title"))
    if not (3 <= len(title) <= a.max_title_len):
        return "title_length"
    ings = [i for i in get_ingredients(rec) if clean_text(i.get("name"))]
    if not (a.min_ingredients <= len(ings) <= a.max_ingredients):
        return "ingredient_count"
    steps = get_steps(rec)
    if not (a.min_steps <= len(steps) <= a.max_steps):
        return "step_count"
    text = " ".join(steps)
    if not (a.min_steps_chars <= len(text) <= a.max_steps_chars):
        return "steps_length"
    if len(text) / len(steps) < a.min_avg_step_chars:
        return "steps_too_terse"
    if re.search(r"https?://|www\.", text):
        return "has_url"
    if cyrillic_ratio(text) < 0.8:
        return "not_russian"
    named = [i for i in ings if not is_staple(i["name"])]
    if named:
        usage = sum(mentions(text, i["name"]) for i in named) / len(named)
        if usage < a.min_ingredient_usage:
            return "ingredients_not_used_in_steps"
    return None


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", default=os.path.join(ROOT, "data/povarenok/data/*.parquet"))
    ap.add_argument("--likes", default=os.path.join(ROOT, "data/povarenok_likes/**/*.parquet"),
                    help="dim/povarenok parquet (likes/ups); skipped if missing")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data/clean"))
    ap.add_argument("--max_candidates", type=int, default=24000)
    ap.add_argument("--max_title_len", type=int, default=90)
    ap.add_argument("--min_ingredients", type=int, default=3)
    ap.add_argument("--max_ingredients", type=int, default=25)
    ap.add_argument("--min_steps", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--min_steps_chars", type=int, default=250)
    ap.add_argument("--max_steps_chars", type=int, default=4000)
    ap.add_argument("--min_avg_step_chars", type=int, default=30)
    ap.add_argument("--min_ingredient_usage", type=float, default=0.4,
                    help="share of (non-staple) ingredients that must be mentioned in the steps")
    ap.add_argument("--mlflow", action="store_true", help="log stats to MLflow experiment 'recipe-data'")
    return ap


def main():
    a = make_parser().parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = read_parquets(a.recipes)
    stats = {"raw": len(df)}
    df["recipe_id"] = df["page_url"].map(recipe_id)

    if glob.glob(a.likes, recursive=True):
        lk = read_parquets(a.likes)
        lk["recipe_id"] = lk["link"].map(recipe_id)
        lk = lk.dropna(subset=["recipe_id"]).drop_duplicates("recipe_id")[["recipe_id", "likes", "ups"]]
        df = df.merge(lk, on="recipe_id", how="left")
        stats["with_likes"] = int(df["likes"].notna().sum())
    else:
        print(f"[warn] no likes data at {a.likes} - popularity = views only")
        df["likes"], df["ups"] = float("nan"), float("nan")

    recs = df.to_dict("records")
    reasons = Counter()
    kept = []
    for r in recs:
        why = reject_reason(r, a)
        reasons[why or "kept"] += 1
        if not why:
            kept.append(r)
    stats.update({f"reject/{k}": v for k, v in reasons.items() if k != "kept"})
    stats["after_filters"] = len(kept)

    out = pd.DataFrame(kept)
    out["views"] = out["views"].fillna(0)
    likes = out["likes"].fillna(0)
    # Popularity: views, plus a bonus for recipes people bothered to like.
    out["popularity"] = out["views"].map(lambda v: math.log1p(v)) + 0.5 * likes.map(lambda v: math.log1p(v))
    out["title_norm"] = out["title"].map(normalize_title)
    out = out.sort_values("popularity", ascending=False).drop_duplicates("title_norm")
    stats["after_dedupe"] = len(out)
    out = out.head(a.max_candidates)
    stats["candidates"] = len(out)

    cols = [c for c in KEEP + ["popularity"] if c in out.columns]
    path = os.path.join(a.out_dir, "candidates.parquet")
    out[cols].to_parquet(path, index=False)
    with open(os.path.join(a.out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"-> {path}")

    if a.mlflow:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("recipe-data")
        with mlflow.start_run(run_name="build_dataset"):
            mlflow.log_params({k: v for k, v in vars(a).items() if k not in ("mlflow",)})
            mlflow.log_metrics({k.replace("/", "_"): v for k, v in stats.items()})


if __name__ == "__main__":
    main()
