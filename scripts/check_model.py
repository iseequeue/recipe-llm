#!/usr/bin/env python
"""Sanity check of a model's chat format before a long training run.

  python scripts/check_model.py --model yandex/YandexGPT-5-Lite-8B-instruct            # tokenizer only, CPU
  srun -p gpu_long --gres=gpu:1 --time=00:20:00 \\
      python scripts/check_model.py --model yandex/YandexGPT-5-Lite-8B-instruct --generate   # + base model answers

Shows how the prompt is rendered, whether the template has a system role, how newlines are
encoded, and (with --generate) the raw and decoded answers of the base model.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache"))

from recipe_llm.chat import ChatFormatter, resolve_newline_token  # noqa: E402
from recipe_llm.prompts import SYSTEM_PROMPT  # noqa: E402

PROMPTS = ["Напиши рецепт: Борщ", "У меня есть: творог, яйца и мука. Что можно приготовить? Напиши рецепт."]
ANSWER = "# Сырники\n\n## Ингредиенты\n- Творог — 400 г\n\n## Приготовление\n1. Смешать.\n2. Обжарить."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--newline_token", default="auto")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    nl = resolve_newline_token(args.newline_token, args.model)
    fmt = ChatFormatter(tok, SYSTEM_PROMPT, nl)
    print(f"tokenizer: {type(tok).__name__}  bos={tok.bos_token!r} eos={tok.eos_token!r} "
          f"pad={tok.pad_token!r} unk={tok.unk_token!r} vocab={len(tok)}")
    print(f"system role supported: {fmt.system_supported}   newline_token: {nl!r}   pad_id used: {fmt.pad_id}")
    print("\n--- chat template (first 800 chars) ---\n" + (tok.chat_template or "<none>")[:800])
    print("\n--- rendered prompt ---\n" + repr(fmt.prompt_text(PROMPTS[0])))
    ids = fmt.prompt_ids(PROMPTS[0])
    print(f"\nprompt ids ({len(ids)}): first {ids[:8]} ... last {ids[-8:]}")
    print("last prompt tokens:", tok.convert_ids_to_tokens(ids[-8:]))
    comp = fmt.completion_ids(ANSWER)
    print(f"\ncompletion: {len(comp)} tokens, ends with eos: {comp[-1] == tok.eos_token_id}")
    print("first completion tokens:", tok.convert_ids_to_tokens(comp[:16]))
    print("round trip ok:", fmt.decode(comp) == ANSWER.strip(), "|", repr(fmt.decode(comp)[:80]))

    if not args.generate:
        return
    import torch

    from recipe_llm.generation import GenParams, generate, load_model

    model, _, fmt = load_model(args.model, None, "fp16", None, args.newline_token, SYSTEM_PROMPT)
    raw_fmt = ChatFormatter(fmt.tok, SYSTEM_PROMPT, fmt.nl)
    for r in generate(model, fmt, PROMPTS, GenParams(max_new_tokens=args.max_new_tokens)):
        p_ids = raw_fmt.prompt_ids(r["prompt"])
        print(f"\n===== {r['prompt']}  ({r['n_tokens']} tokens, truncated={r['truncated']})\n{r['response']}")
    with torch.no_grad():
        ids = torch.tensor([p_ids], device=next(model.parameters()).device)
        out = model.generate(input_ids=ids, max_new_tokens=60, do_sample=False, pad_token_id=fmt.pad_id)
    print("\n--- raw output (special tokens kept, first 60 tokens) ---\n" + repr(fmt.tok.decode(out[0, ids.shape[1]:])))


if __name__ == "__main__":
    main()
