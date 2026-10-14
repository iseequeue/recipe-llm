"""Training config: YAML file + `key=value` CLI overrides."""
from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml

from .prompts import SYSTEM_PROMPT


@dataclass
class TrainConfig:
    # --- model ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # V100 has fp16 tensor cores but no bf16: keep fp16 (fp32 weights work too, AMP is used either way).
    model_dtype: str = "fp16"  # fp16 | fp32 | bf16 (bf16 only on Ampere+)
    attn_implementation: str = "sdpa"  # flash-attention 2 needs Ampere+; SDPA uses mem-efficient kernels on V100
    # "auto": "[NL]" for YandexGPT (it was trained with newlines replaced by [NL]), none otherwise
    newline_token: Optional[str] = "auto"
    system_prompt: str = SYSTEM_PROMPT  # folded into the user turn for models without a system role

    # --- data ---
    dataset_name: str = "d0rj/povarenok_recipes_detail"
    dataset_split: str = "train"
    data_files: Optional[str] = None  # local parquet/json(l) instead of the Hub dataset
    max_train_samples: int = 20000
    max_eval_samples: int = 300
    # take the best rows first by this column ("judge_score" after scripts/score_recipes.py,
    # "views" for the raw dataset); None = random
    sort_by: Optional[str] = "views"
    include_description: bool = True  # user-written descriptions are mostly fluff ("Очень вкусно!")
    max_seq_len: int = 1024  # longer examples are dropped, not truncated
    prompt_mix: dict = field(default_factory=lambda: {"title": 0.5, "ingredients": 0.3, "category": 0.2})
    preprocessing_num_workers: int = 4

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # --- optimisation ---
    output_dir: str = "outputs/qwen2.5-1.5b-recipes"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    dataloader_num_workers: int = 2
    seed: int = 42

    # --- logging / checkpoints ---
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    eval_on_start: bool = True
    resume_from_checkpoint: Optional[str] = None

    # --- MLflow ---
    mlflow_tracking_uri: Optional[str] = None  # None -> $MLFLOW_TRACKING_URI or sqlite:///mlflow.db
    mlflow_experiment: str = "recipe-llm"
    run_name: Optional[str] = None
    log_adapter_to_mlflow: bool = True

    # --- sample generations logged to MLflow at every eval ---
    sample_prompts: list = field(
        default_factory=lambda: [
            "Напиши рецепт: Борщ с пампушками",
            "У меня есть: куриное филе, шампиньоны и сливки. Что можно приготовить? Напиши рецепт.",
            "Предложи рецепт из раздела «Выпечка сладкая».",
        ]
    )
    sample_max_new_tokens: int = 384

    # --- debug ---
    use_cpu: bool = False


def _parse_value(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        low = raw.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null"):
            return None
        return raw


def load_config(path: Optional[str], overrides: list[str]) -> TrainConfig:
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        k, v = item.split("=", 1)
        data[k.strip()] = _parse_value(v.strip())
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainConfig(**data)
