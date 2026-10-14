"""Dataset loading, cleaning and tokenisation with prompt masking."""
from __future__ import annotations

import random

import numpy as np
from datasets import Dataset, load_dataset

from .chat import ChatFormatter, resolve_newline_token
from .config import TrainConfig
from .prompts import format_recipe, is_usable, make_user_prompt

KEEP_COLUMNS = [
    "title", "description", "views", "likes", "ups", "judge_score", "kroshki", "ingredients",
    "cooking_time", "portions_count", "cooking_steps", "purposes", "tastes",
]


def make_formatter(cfg: TrainConfig, tokenizer) -> ChatFormatter:
    return ChatFormatter(tokenizer, cfg.system_prompt, resolve_newline_token(cfg.newline_token, cfg.model_name))


def load_raw(cfg: TrainConfig) -> Dataset:
    if cfg.data_files:
        ext = cfg.data_files.rsplit(".", 1)[-1].lower()
        builder = {"jsonl": "json", "json": "json", "parquet": "parquet", "csv": "csv"}[ext]
        ds = load_dataset(builder, data_files=cfg.data_files, split="train")
    else:
        ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    return ds.remove_columns([c for c in ds.column_names if c not in KEEP_COLUMNS])


def encode_example(rec: dict, fmt: ChatFormatter, prompt_type: str, user_text: str,
                   include_description: bool = True) -> dict:
    """Prompt = the model's own chat template (as at inference); completion = answer + EOS.
    Loss is computed on the completion only."""
    prompt_ids = fmt.prompt_ids(user_text)
    completion_ids = fmt.completion_ids(format_recipe(rec, include_description))
    input_ids = prompt_ids + completion_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "length": len(input_ids),
        "prompt_type": prompt_type,
        "user_text": user_text,
    }


def build_datasets(cfg: TrainConfig, tokenizer, log=print) -> tuple[Dataset, Dataset, dict]:
    raw = load_raw(cfg)
    n_raw = len(raw)
    ds = raw.filter(is_usable, num_proc=cfg.preprocessing_num_workers, desc="filter unusable")
    n_usable = len(ds)

    if cfg.sort_by and cfg.sort_by in ds.column_names:
        ds = ds.sort(cfg.sort_by, reverse=True)
    else:
        if cfg.sort_by:
            log(f"sort_by={cfg.sort_by!r} not in columns {ds.column_names} -> random order")
        ds = ds.shuffle(seed=cfg.seed)
    # Take a margin on top of what we need: some examples are dropped by length.
    want = cfg.max_train_samples + cfg.max_eval_samples
    if want > 0:
        ds = ds.select(range(min(len(ds), int(want * 1.3) + 16)))

    mix = dict(cfg.prompt_mix)
    seed = cfg.seed
    fmt = make_formatter(cfg, tokenizer)
    with_desc = cfg.include_description
    log(f"Chat format: system_role={fmt.system_supported} newline_token={fmt.nl!r}\n"
        f"Prompt example:\n{fmt.prompt_text('Напиши рецепт: Сырники')!r}")

    def _map(rec, idx):
        rng = random.Random(seed * 1_000_003 + idx)
        prompt_type, user_text = make_user_prompt(rec, rng, mix)
        return encode_example(rec, fmt, prompt_type, user_text, with_desc)

    tok = ds.map(
        _map, with_indices=True, remove_columns=ds.column_names,
        num_proc=cfg.preprocessing_num_workers, desc="tokenize",
    )
    n_before_len = len(tok)
    tok = tok.filter(lambda x: x <= cfg.max_seq_len, input_columns="length", desc="length filter")
    n_after_len = len(tok)

    tok = tok.shuffle(seed=cfg.seed)
    n_eval = min(cfg.max_eval_samples, max(1, len(tok) // 20))
    eval_ds = tok.select(range(n_eval))
    train_ds = tok.select(range(n_eval, min(len(tok), n_eval + cfg.max_train_samples)))

    lengths = np.array(train_ds["length"])
    types = train_ds["prompt_type"]
    stats = {
        "data/raw_rows": n_raw,
        "data/usable_rows": n_usable,
        "data/dropped_by_length_frac": round(1 - n_after_len / max(1, n_before_len), 4),
        "data/train_examples": len(train_ds),
        "data/eval_examples": len(eval_ds),
        "data/train_tokens": int(lengths.sum()),
        "data/len_mean": float(lengths.mean()),
        "data/len_p95": float(np.percentile(lengths, 95)),
        "data/len_max": int(lengths.max()),
        **{f"data/prompt_type_{t}": types.count(t) for t in sorted(set(types))},
    }
    log(f"Dataset stats: {stats}")
    return train_ds, eval_ds, stats


def preview_examples(ds: Dataset, tokenizer, n: int = 5) -> str:
    # Decoded without [NL] -> \n conversion on purpose: shows exactly what the model is trained on.
    """Markdown with decoded training examples (useful as an MLflow artifact)."""
    out = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        target = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
        out.append(
            f"## Example {i} ({ex['prompt_type']}, {ex['length']} tokens)\n\n"
            f"**User:** {ex['user_text']}\n\n**Target:**\n\n{tokenizer.decode(target)}\n"
        )
    return "\n---\n".join(out)
