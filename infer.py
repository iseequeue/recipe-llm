#!/usr/bin/env python
"""Recipe generation with a fine-tuned LoRA adapter (or a base model).

  python infer.py --adapter outputs/yandexgpt5-recipes/final_adapter --prompt "Напиши рецепт: Сырники"
  python infer.py --adapter ... --ingredients "кабачки, фарш, сыр"
  python infer.py --adapter ... --interactive
  python infer.py --adapter ... --input_file prompts.txt --output_file out.jsonl --batch_size 8
  python infer.py --adapter runs:/<mlflow_run_id>/adapter --prompt "..."          # adapter from MLflow
  python infer.py --base_model yandex/YandexGPT-5-Lite-8B-instruct --prompt "..."  # base model, no LoRA
  python infer.py --adapter ... --merge_to merged_model/                           # save merged model
  python infer.py --adapter ... --rag data/rag_index --prompt "Напиши рецепт: Солянка"  # with retrieved recipes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe_llm.generation import DTYPES, GenParams, generate, load_model, stream  # noqa: E402
from recipe_llm.prompts import ingredients_to_prompt  # noqa: E402
from recipe_llm.rag import load_retriever  # noqa: E402


def print_stream(model, fmt, prompt, params):
    shown = ""
    for text in stream(model, fmt, prompt, params):
        print(text[len(shown):] if text.startswith(shown) else "\n" + text, end="", flush=True)
        shown = text
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="LoRA adapter dir or MLflow URI (runs:/<id>/adapter)")
    ap.add_argument("--base_model", help="default: taken from adapter_config.json")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--prompt", help="free-form request, e.g. 'Напиши рецепт: Сырники'")
    g.add_argument("--ingredients", help="comma-separated list of products you have")
    g.add_argument("--input_file", help="text file, one request per line")
    g.add_argument("--interactive", action="store_true")
    g.add_argument("--merge_to", help="merge LoRA into the base model and save it to this dir")
    ap.add_argument("--output_file", help="jsonl output for --input_file")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.5, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES), help="V100: fp16")
    ap.add_argument("--device", default=None, help="cuda | cuda:1 | cpu | auto (split over GPUs)")
    ap.add_argument("--newline_token", default="auto", help="'auto' ([NL] for YandexGPT), 'none' or a token")
    ap.add_argument("--system_prompt", default=None, help="default: the one used in training")
    ap.add_argument("--rag", help="RAG index dir: put similar recipes from the base into the prompt")
    ap.add_argument("--rag_k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer, fmt = load_model(args.base_model, args.adapter, args.dtype, args.device,
                                       args.newline_token, args.system_prompt)
    params = GenParams(args.max_new_tokens, args.temperature, args.top_p, args.repetition_penalty)
    retriever = load_retriever(args.rag, args.device)

    def with_rag(prompt: str) -> str:
        if not retriever:
            return prompt
        aug, hits = retriever.augment(prompt, args.rag_k)
        print("[RAG] " + " | ".join(f"{h['title']} ({h['similarity']:.2f})" for h in hits), file=sys.stderr)
        return aug

    if args.merge_to:
        model.save_pretrained(args.merge_to)
        tokenizer.save_pretrained(args.merge_to)
        print(f"Merged model saved to {args.merge_to}")
        return

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        out_f = open(args.output_file, "w", encoding="utf-8") if args.output_file else None
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            t0 = time.time()
            for p, r in zip(batch, generate(model, fmt, [with_rag(x) for x in batch], params)):
                r["prompt"] = p
                if out_f:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                else:
                    print(f"\n### {r['prompt']}\n\n{r['response']}\n")
            print(f"[{i + len(batch)}/{len(prompts)}] {time.time() - t0:.1f}s", file=sys.stderr)
        if out_f:
            out_f.close()
        return

    if args.interactive:
        print("Введите блюдо, или 'из: продукт1, продукт2', или любой запрос. Пустая строка — выход.")
        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                break
            if not line:
                break
            if line.lower().startswith("из:"):
                line = ingredients_to_prompt(line[3:].split(","))
            elif len(line.split()) <= 4 and "?" not in line:
                line = f"Напиши рецепт: {line}"
            print()
            print_stream(model, fmt, with_rag(line), params)
        return

    prompt = ingredients_to_prompt(args.ingredients.split(",")) if args.ingredients else args.prompt
    if not prompt:
        ap.error("Give --prompt, --ingredients, --input_file, --interactive or --merge_to")
    print_stream(model, fmt, with_rag(prompt), params)


if __name__ == "__main__":
    main()
