"""Model-agnostic chat encoding shared by training, evaluation, scoring and the demo.

Why this exists: models disagree on chat formats.
* Qwen uses ChatML with a system role.
* YandexGPT-5-Lite uses " Пользователь: ... Ассистент:[SEP]", has no system role and
  was trained with newlines replaced by the literal "[NL]".

So instead of rendering a whole conversation as text and slicing it, we
  * build the prompt with the model's own chat template (tokenize=True), exactly as the
    model authors do at inference time;
  * encode the answer separately and append EOS - the same tokens the model must learn
    to produce after the generation prompt.
Training and inference go through the same object, so they can never drift apart.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

_SYSTEM_MARKER = "SYSTEM_PROMPT_MARKER_7f3a"


def default_newline_token(model_name: str) -> Optional[str]:
    """YandexGPT-5 was trained with '\\n' replaced by '[NL]' (see its model card)."""
    return "[NL]" if "yandexgpt" in (model_name or "").lower() else None


def resolve_newline_token(value: Optional[str], model_name: str) -> Optional[str]:
    if value in (None, "", "none", "None", False):
        return None
    if value == "auto":
        return default_newline_token(model_name)
    return value


def _ids(out) -> list[int]:
    """apply_chat_template returns a list (transformers 4.x) or a BatchEncoding (5.x)."""
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


class ChatFormatter:
    def __init__(self, tokenizer, system_prompt: Optional[str] = None, newline_token: Optional[str] = None):
        self.tok = tokenizer
        self.system_prompt = system_prompt or None
        self.nl = newline_token or None
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no eos_token")
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = next(
            i for i in (tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.eos_token_id) if i is not None
        )
        self.system_supported = self._check_system_role()
        self._nl_re = re.compile(r" ?" + re.escape(self.nl) + r" ?") if self.nl else None

    # ---- text transforms -------------------------------------------------
    def encode_text(self, text: str) -> str:
        return text.replace("\n", self.nl) if self.nl else text

    def decode_text(self, text: str) -> str:
        return self._nl_re.sub("\n", text) if self.nl else text

    # ---- conversation ----------------------------------------------------
    def _check_system_role(self) -> bool:
        template = getattr(self.tok, "chat_template", None) or ""
        if "system" not in template:
            return False
        try:
            text = self.tok.apply_chat_template(
                [{"role": "system", "content": _SYSTEM_MARKER}, {"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return False
        return _SYSTEM_MARKER in text

    def messages(self, user_text: str) -> list[dict]:
        user = self.encode_text(user_text)
        if not self.system_prompt:
            return [{"role": "user", "content": user}]
        system = self.encode_text(self.system_prompt)
        if self.system_supported:
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # No system role (YandexGPT): put the instruction in front of the request.
        return [{"role": "user", "content": system + self.encode_text("\n\n") + user}]

    def prompt_text(self, user_text: str) -> str:
        return self.tok.apply_chat_template(self.messages(user_text), tokenize=False, add_generation_prompt=True)

    def prompt_ids(self, user_text: str) -> list[int]:
        return _ids(self.tok.apply_chat_template(self.messages(user_text), tokenize=True, add_generation_prompt=True))

    def completion_ids(self, answer: str) -> list[int]:
        ids = self.tok(self.encode_text(answer), add_special_tokens=False)["input_ids"]
        return list(ids) + [self.eos_id]

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_text(self.tok.decode(list(ids), skip_special_tokens=True)).strip()

    def left_pad(self, batch: list[list[int]]):
        import torch

        width = max(map(len, batch))
        input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            mask[i, width - len(ids):] = 1
        return input_ids, mask
