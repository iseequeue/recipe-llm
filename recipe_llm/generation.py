"""Loading a (base + LoRA) model and generating recipes. Used by infer.py, evaluate.py,
the judge scorer, the Gradio app and sample generation during training."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from threading import Thread
from typing import Iterator, Optional

import torch
import yaml

from .chat import ChatFormatter, resolve_newline_token
from .prompts import SYSTEM_PROMPT

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


@dataclass
class GenParams:
    max_new_tokens: int = 768
    temperature: float = 0.0  # 0 = greedy
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    def kwargs(self, pad_id: int, eos_id: int) -> dict:
        kw = dict(max_new_tokens=self.max_new_tokens, repetition_penalty=self.repetition_penalty,
                  pad_token_id=pad_id, eos_token_id=eos_id, do_sample=self.temperature > 0,
                  use_cache=True)
        if self.temperature > 0:
            kw.update(temperature=self.temperature, top_p=self.top_p)
        return kw


def resolve_adapter(path: Optional[str]) -> Optional[str]:
    if path and path.split(":", 1)[0] in ("runs", "models", "mlflow-artifacts"):
        import mlflow

        os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        return mlflow.artifacts.download_artifacts(artifact_uri=path)
    return path


def _training_cfg(adapter: Optional[str]) -> dict:
    if adapter:
        p = os.path.join(adapter, "recipe_training_config.yaml")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def load_model(base_model: Optional[str] = None, adapter: Optional[str] = None, dtype: str = "fp16",
               device: Optional[str] = None, newline_token: Optional[str] = "auto",
               system_prompt: Optional[str] = None, merge: bool = True):
    """Returns (model, tokenizer, formatter). Settings saved with the adapter at training
    time (system prompt, newline handling) take precedence, so inference matches training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = resolve_adapter(adapter)
    tcfg = _training_cfg(adapter)
    if adapter and not base_model:
        with open(os.path.join(adapter, "adapter_config.json"), encoding="utf-8") as f:
            base_model = json.load(f)["base_model_name_or_path"]
    if not base_model:
        raise ValueError("Give base_model and/or adapter")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = DTYPES[dtype] if device != "cpu" else torch.float32
    tok_src = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch_dtype, attn_implementation="sdpa",
        device_map="auto" if device == "auto" else None,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    if device != "auto":
        model.to(device)
    model.eval()

    nl = resolve_newline_token(tcfg.get("newline_token", newline_token), tcfg.get("model_name", base_model))
    sp = system_prompt if system_prompt is not None else tcfg.get("system_prompt", SYSTEM_PROMPT)
    fmt = ChatFormatter(tokenizer, sp, nl)
    print(f"Loaded base={base_model} adapter={adapter} device={device} dtype={torch_dtype} "
          f"system_role={fmt.system_supported} newline_token={nl!r}", file=sys.stderr)
    return model, tokenizer, fmt


@torch.no_grad()
def generate(model, fmt: ChatFormatter, prompts: list[str], params: GenParams) -> list[dict]:
    """Batched generation. Returns [{prompt, response, n_tokens, truncated}]."""
    ids = [fmt.prompt_ids(p) for p in prompts]
    input_ids, mask = fmt.left_pad(ids)
    device = next(model.parameters()).device
    out = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                         **params.kwargs(fmt.pad_id, fmt.eos_id))
    res = []
    for p, row in zip(prompts, out[:, input_ids.shape[1]:].tolist()):
        n = next((i + 1 for i, t in enumerate(row) if t == fmt.eos_id), len(row))
        res.append({"prompt": p, "response": fmt.decode(row[:n]), "n_tokens": n,
                    "truncated": fmt.eos_id not in row[:n]})
    return res


def stream(model, fmt: ChatFormatter, prompt: str, params: GenParams, use_adapter: bool = True) -> Iterator[str]:
    """Yields the growing response text (for CLI streaming and the Gradio app).
    use_adapter=False answers with the base model (only for an unmerged PeftModel)."""
    from transformers import TextIteratorStreamer

    input_ids, mask = fmt.left_pad([fmt.prompt_ids(prompt)])
    device = next(model.parameters()).device
    streamer = TextIteratorStreamer(fmt.tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(input_ids=input_ids.to(device), attention_mask=mask.to(device), streamer=streamer,
              **params.kwargs(fmt.pad_id, fmt.eos_id))
    def run():
        with torch.no_grad():
            if not use_adapter and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    model.generate(**kw)
            else:
                model.generate(**kw)

    thread = Thread(target=run, daemon=True)
    thread.start()
    raw = ""
    for chunk in streamer:
        raw += chunk
        yield fmt.decode_text(raw).strip()
    thread.join()
