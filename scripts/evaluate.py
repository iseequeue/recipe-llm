#!/usr/bin/env python
"""Generate answers for the fixed evaluation set and compute automatic metrics.

  python scripts/evaluate.py --name yandex-base --base_model yandex/YandexGPT-5-Lite-8B-instruct
  python scripts/evaluate.py --name yandex-ft   --adapter outputs/yandexgpt5-recipes/final_adapter
  python scripts/evaluate.py --name qwen-ft     --adapter outputs/qwen2.5-7b-clean-recipes/final_adapter
  python scripts/evaluate.py --name yandex-ft --rescore          # recompute metrics, no generation
  python scripts/evaluate.py --name yandex-ft --adapter ... --resume   # generate only prompts missing in the jsonl

-> eval/results/<name>.jsonl, eval/results/<name>.summary.json, MLflow experiment "recipe-eval".
Decoding is greedy (temperature 0) so that systems are compared on equal terms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from recipe_llm.metrics import recipe_metrics, summarize  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="system name, e.g. yandex-base / yandex-ft")
    ap.add_argument("--adapter")
    ap.add_argument("--base_model")
    ap.add_argument("--prompts", default=os.path.join(ROOT, "eval/prompts.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "eval/results"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--newline_token", default="auto")
    ap.add_argument("--rag", help="RAG index dir (scripts/build_rag_index.py): prepend similar recipes to each prompt")
    ap.add_argument("--rag_k", type=int, default=3)
    ap.add_argument("--rescore", action="store_true", help="only recompute metrics from the existing jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="keep answers already in the jsonl (same prompt id and text) and generate only the missing ones; "
                         "decoding is greedy, so old answers stay valid. Progress is saved after every batch.")
    ap.add_argument("--no_mlflow", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.name}.jsonl")
    prompts = load_jsonl(a.prompts)

    summary_path = os.path.join(a.out_dir, f"{a.name}.summary.json")
    if a.rescore:
        rows = load_jsonl(out_path)
        if os.path.exists(summary_path):  # keep what was evaluated
            with open(summary_path, encoding="utf-8") as f:
                old = json.load(f)
            a.adapter, a.base_model, a.rag = old.get("adapter"), old.get("base_model"), old.get("rag")
    else:
        from recipe_llm.generation import GenParams, generate, load_model
        from recipe_llm.rag import load_retriever

        model, _, fmt = load_model(a.base_model, a.adapter, a.dtype, a.device, a.newline_token)
        retriever = load_retriever(a.rag, a.device)
        params = GenParams(max_new_tokens=a.max_new_tokens, temperature=0.0, repetition_penalty=a.repetition_penalty)
        done = {}
        if a.resume and os.path.exists(out_path):
            done = {r["id"]: r for r in load_jsonl(out_path)}
            done = {p["id"]: done[p["id"]] for p in prompts
                    if p["id"] in done and done[p["id"]].get("prompt") == p["prompt"]}
            print(f"resume: {len(done)} of {len(prompts)} answers already in {out_path}")
        todo = [p for p in prompts if p["id"] not in done]
        rows, t0 = [], time.time()
        for i in range(0, len(todo), a.batch_size):
            batch = todo[i:i + a.batch_size]
            texts, hits = [p["prompt"] for p in batch], [None] * len(batch)
            if retriever:
                aug = [retriever.augment(t, a.rag_k) for t in texts]
                texts, hits = [x[0] for x in aug], [x[1] for x in aug]
            for p, g, h in zip(batch, generate(model, fmt, texts, params), hits):
                row = {**p, "system": a.name, "response": g["response"], "n_tokens": g["n_tokens"],
                       "truncated": g["truncated"]}
                if h is not None:
                    row["retrieved"] = [{k: x[k] for k in ("recipe_id", "title", "similarity")} for x in h]
                    row["retrieved_top1"] = h[0]["recipe_md"] if h else ""
                rows.append(row)
            print(f"[{len(rows)}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
            if a.resume:  # checkpoint: a job killed by the time limit continues from here
                part = {**done, **{r["id"]: r for r in rows}}
                with open(out_path + ".tmp", "w", encoding="utf-8") as f:
                    for p in prompts:
                        if p["id"] in part:
                            f.write(json.dumps(part[p["id"]], ensure_ascii=False) + "\n")
                os.replace(out_path + ".tmp", out_path)
        new = {r["id"]: r for r in rows}
        rows = [done.get(p["id"]) or new[p["id"]] for p in prompts]

    # Prompt metadata (must_have, requested) always comes from the current prompts file.
    meta = {p["id"]: p for p in prompts}
    rows = [{**r, **meta.get(r["id"], {})} for r in rows]
    for r in rows:
        r["metrics"] = recipe_metrics(r["response"], r.get("must_have"), r.get("requested"), r.get("truncated"))
        if "retrieved_top1" in r and (r.get("must_have") or r.get("requested")):
            # How good was the search itself: does the top retrieved recipe have the key / requested ingredients?
            rm = recipe_metrics(r["retrieved_top1"], r.get("must_have"), r.get("requested"))
            r["metrics"]["retrieval_quality"] = rm.get("must_have_recall", rm.get("requested_coverage"))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"system": a.name, "adapter": a.adapter, "base_model": a.base_model, "rag": a.rag, **summarize(rows)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not a.no_mlflow:
        try:
            log_to_mlflow(a, rows, summary, out_path)
        except Exception as e:  # results are already on disk; MLflow is a convenience
            print(f"[warn] MLflow logging failed: {e}")


def log_to_mlflow(a, rows, summary, out_path):
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("recipe-eval")
        with mlflow.start_run(run_name=a.name):
            mlflow.log_params({"system": a.name, "adapter": a.adapter, "base_model": a.base_model, "rag": a.rag,
                               "n_prompts": len(rows), "decoding": "greedy",
                               "repetition_penalty": a.repetition_penalty})
            mlflow.log_metrics({k.replace("/", "."): v for k, v in summary.items() if isinstance(v, (int, float))})
            mlflow.log_table({"id": [r["id"] for r in rows], "type": [r["type"] for r in rows],
                              "prompt": [r["prompt"] for r in rows], "response": [r["response"] for r in rows]},
                             "generations.json")
            mlflow.log_artifact(out_path)


if __name__ == "__main__":
    main()
