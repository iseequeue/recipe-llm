#!/usr/bin/env python
"""LoRA fine-tuning of an instruct LLM for recipe generation, with MLflow logging.

1 GPU:   python train.py --config configs/qwen2.5-1.5b_v100.yaml
2 GPUs:  torchrun --nproc_per_node=2 train.py --config configs/qwen2.5-1.5b_v100.yaml
Overrides: python train.py --config ... max_train_samples=5000 learning_rate=1e-4
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import platform

import mlflow
import peft
import torch
import transformers
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from recipe_llm.config import load_config
from recipe_llm.data import build_datasets, make_formatter, preview_examples
from recipe_llm.generation import GenParams, generate

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log0(*a):
    if is_main():
        print(*a, flush=True)


def check_gpu(cfg) -> None:
    if cfg.use_cpu or not torch.cuda.is_available():
        log0("CUDA not available -> running on CPU (debug only)")
        return
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i)
        log0(f"GPU {i}: {torch.cuda.get_device_name(i)} cc={cc[0]}.{cc[1]}")
    arch = torch.cuda.get_arch_list()
    cc = torch.cuda.get_device_capability(0)
    sm = cc[0] * 10 + cc[1]
    has_sass = f"sm_{sm}" in arch
    has_ptx = any(a.startswith("compute_") and int(a.split("_")[1]) <= sm for a in arch)
    if not (has_sass or has_ptx):
        raise RuntimeError(
            f"This PyTorch build has no kernels for sm_{sm} (has {arch}). "
            "V100 (sm_70) / P100 (sm_60) need the cu126 build: "
            "pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126"
        )
    # Native bf16 starts with Ampere (sm_80); on V100/P100 torch may report emulated support.
    if cfg.model_dtype == "bf16" and cc < (8, 0):
        raise RuntimeError(f"No native bf16 on sm_{sm} (V100/P100) - use model_dtype=fp16")
    if cfg.attn_implementation == "flash_attention_2" and cc < (8, 0):
        raise RuntimeError("flash-attention 2 needs Ampere+; use attn_implementation=sdpa on V100")


@torch.no_grad()
def generate_samples(model, fmt, prompts, max_new_tokens) -> list[dict]:
    """Greedy generations for a fixed set of prompts (logged to MLflow at every eval)."""
    was_training = model.training
    model.eval()
    out = []
    for p in prompts:  # one at a time: robust to the memory left over by training
        out += generate(model, fmt, [p], GenParams(max_new_tokens=max_new_tokens, temperature=0.0))
    if was_training:
        model.train()
    return out


def samples_to_md(samples, title) -> str:
    body = "\n\n---\n\n".join(f"### {s['prompt']}\n\n{s['response']}" for s in samples)
    return f"# {title}\n\n{body}\n"


class MlflowExtrasCallback(TrainerCallback):
    """Perplexity next to eval_loss + greedy sample generations at each evaluation."""

    def __init__(self, fmt, cfg):
        self.fmt, self.cfg = fmt, cfg

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
        if not state.is_world_process_zero or mlflow.active_run() is None:
            return
        if metrics and "eval_loss" in metrics:
            mlflow.log_metric("eval_perplexity", math.exp(min(metrics["eval_loss"], 20)), step=state.global_step)
        if self.cfg.sample_prompts and model is not None:
            samples = generate_samples(model, self.fmt, self.cfg.sample_prompts, self.cfg.sample_max_new_tokens)
            mlflow.log_text(samples_to_md(samples, f"Step {state.global_step}"), f"samples/step_{state.global_step:06d}.md")


def make_training_args(**kw) -> TrainingArguments:
    """Works across transformers 4.4x and 5.x (renamed/removed arguments)."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in kw and "warmup_ratio" not in params:  # v5: float warmup_steps < 1 == ratio
        kw["warmup_steps"] = kw.pop("warmup_ratio")
    if "group_by_length" in kw and "group_by_length" not in params:
        kw["train_sampling_strategy"] = "group_by_length" if kw.pop("group_by_length") else "random"
    return TrainingArguments(**kw)


def flatten(d: dict) -> dict:
    return {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value overrides of config fields")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)
    check_gpu(cfg)

    on_gpu = torch.cuda.is_available() and not cfg.use_cpu
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # MLflowCallback inside Trainer reads these env vars, so set them for all ranks.
    tracking_uri = cfg.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow_experiment
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "0")  # we log the adapter ourselves

    targs = make_training_args(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        fp16=on_gpu and cfg.model_dtype in ("fp16", "fp32"),  # fp16 AMP on tensor cores; V100 has no bf16
        bf16=on_gpu and cfg.model_dtype == "bf16",
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=cfg.group_by_length,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        eval_on_start=cfg.eval_on_start,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["mlflow"],
        run_name=cfg.run_name,
        dataloader_num_workers=cfg.dataloader_num_workers,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=cfg.seed,
        use_cpu=not on_gpu,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with targs.main_process_first(desc="dataset preprocessing"):
        train_ds, eval_ds, data_stats = build_datasets(cfg, tokenizer, log=log0)
    preview_md = preview_examples(train_ds, tokenizer) if is_main() else ""
    cols = ["input_ids", "attention_mask", "labels"]
    train_ds, eval_ds = train_ds.select_columns(cols), eval_ds.select_columns(cols)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=DTYPES[cfg.model_dtype] if on_gpu else torch.float32,
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)  # LoRA weights are kept in fp32 -> safe with fp16 AMP
    trainable, total = model.get_nb_trainable_parameters()
    log0(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
        callbacks=[MlflowExtrasCallback(make_formatter(cfg, tokenizer), cfg)],
    )

    if is_main():
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run(run_name=cfg.run_name)
        mlflow.log_params({f"cfg.{k}": v for k, v in flatten(dataclasses.asdict(cfg)).items()})
        mlflow.log_params({
            "trainable_params": trainable, "total_params": total, "world_size": world_size,
            "effective_batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size,
        })
        mlflow.set_tags({
            "base_model": cfg.model_name, "dataset": cfg.data_files or cfg.dataset_name,
            "gpu": torch.cuda.get_device_name(0) if on_gpu else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "python": platform.python_version(),
        })
        mlflow.log_metrics(data_stats, step=0)
        mlflow.log_text(yaml.safe_dump(dataclasses.asdict(cfg), allow_unicode=True, sort_keys=False), "config.yaml")
        mlflow.log_text(preview_md, "data_preview.md")
        mlflow.log_text(cfg.system_prompt, "system_prompt.txt")
        log0(f"MLflow run: {mlflow.active_run().info.run_id} @ {tracking_uri}")

    train_result = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    final_eval = trainer.evaluate()  # best checkpoint (load_best_model_at_end)

    final_dir = os.path.join(cfg.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        with open(os.path.join(final_dir, "recipe_training_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(cfg), f, allow_unicode=True, sort_keys=False)

        mlflow.log_metrics({f"final/{k}": v for k, v in {**train_result.metrics, **final_eval}.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metric("final/eval_perplexity", math.exp(min(final_eval["eval_loss"], 20)))
        if cfg.sample_prompts:
            samples = generate_samples(trainer.model, make_formatter(cfg, tokenizer), cfg.sample_prompts, cfg.sample_max_new_tokens)
            mlflow.log_table({"prompt": [s["prompt"] for s in samples], "response": [s["response"] for s in samples]},
                             "samples/final.json")
            mlflow.log_text(samples_to_md(samples, "Final model"), "samples/final.md")
        if cfg.log_adapter_to_mlflow:
            mlflow.log_artifacts(final_dir, artifact_path="adapter")
        run_id = mlflow.active_run().info.run_id
        mlflow.end_run()
        print(f"\nDone. Adapter: {final_dir}\nMLflow run id: {run_id}\n"
              f"Inference: python infer.py --adapter {final_dir} --prompt 'Напиши рецепт: Сырники'")


if __name__ == "__main__":
    main()
