"""Prompt templates and recipe formatting shared by training and inference.

Training and inference MUST use the same system prompt and chat template,
otherwise the fine-tuned model will see a distribution shift at inference.
"""
from __future__ import annotations

import html
import random
import re
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "Ты — опытный шеф-повар и кулинарный автор. По запросу пользователя пиши "
    "подробный и выполнимый рецепт на русском языке: название, короткое описание, "
    "время приготовления и количество порций, список ингредиентов с количествами "
    "и пронумерованные шаги приготовления."
)

TITLE_TEMPLATES = [
    "Напиши рецепт: {title}",
    "Как приготовить «{title}»? Дай подробный рецепт.",
    "Дай рецепт блюда «{title}».",
]
INGREDIENT_TEMPLATES = [
    "У меня есть: {ings}. Что можно приготовить? Напиши рецепт.",
    "Придумай рецепт, в котором используются {ings}.",
    "Что приготовить из таких продуктов: {ings}? Нужен подробный рецепт.",
]
CATEGORY_TEMPLATES = [
    "Предложи рецепт из раздела «{cat}»{extra}.",
    "Хочу приготовить что-нибудь из категории «{cat}»{extra}. Напиши рецепт.",
]

# Ingredients that say nothing about the dish; skipped when building
# "what can I cook from X, Y, Z" prompts.
_STAPLES = ("соль", "перец", "вода", "сахар", "масло растительное", "специи", "зелень")

_WS_RE = re.compile(r"[ \t ]+")
_NL_RE = re.compile(r"\n{3,}")


def as_list(x: Any) -> list:
    """Lists come back as numpy arrays from pandas and as None/NaN when missing."""
    if x is None or (isinstance(x, float) and x != x):
        return []
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, dict):
        return [x]
    return list(x)


def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = html.unescape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NL_RE.sub("\n\n", text).strip()


def get_ingredients(rec: dict) -> list[dict]:
    """List of {"name", "count"}; also accepts the dict-of-lists layout that
    older `datasets` versions produce for Sequence(struct) features."""
    ings = rec.get("ingredients")
    if hasattr(ings, "tolist"):
        ings = ings.tolist()
    if isinstance(ings, dict) and "name" in ings and not isinstance(ings.get("name"), str):
        names, counts = ings.get("name") or [], ings.get("count") or []
        counts = list(counts) + [None] * (len(names) - len(counts))
        ings = [{"name": n, "count": c} for n, c in zip(names, counts)]
    return [i for i in as_list(ings) if isinstance(i, dict) and i]


def _ingredient_line(ing: dict) -> str | None:
    name = clean_text(ing.get("name"))
    if not name:
        return None
    count = clean_text(ing.get("count"))
    return f"- {name} — {count}" if count else f"- {name}"


_PHOTO_RE = re.compile(
    r"\s*\((?:[^()]*?\bфото\b[^()]*?)\)|\s*,?\s*(?:как|см\.?|смотри(?:те)?)\s+(?:на\s+)?фото\s*\d*", re.IGNORECASE
)


def clean_step(text: Any) -> str:
    """Step text without references to photos that the model cannot see."""
    text = _PHOTO_RE.sub("", clean_text(text))
    return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()


def get_steps(rec: dict) -> list[str]:
    return [s for s in (clean_step(x) for x in as_list(rec.get("cooking_steps"))) if s]


def format_recipe(rec: dict, include_description: bool = True) -> str:
    """Render a povarenok record as the target (assistant) markdown text."""
    parts = [f"# {clean_text(rec['title'])}"]
    desc = clean_text(rec.get("description")) if include_description else ""
    if desc:
        parts.append(desc)

    meta = []
    if clean_text(rec.get("cooking_time")):
        meta.append(f"**Время приготовления:** {clean_text(rec['cooking_time'])}")
    if clean_text(rec.get("portions_count")):
        meta.append(f"**Порций:** {clean_text(rec['portions_count'])}")
    if meta:
        parts.append("  \n".join(meta))

    ing_lines = [l for l in (_ingredient_line(i) for i in get_ingredients(rec)) if l]
    parts.append("## Ингредиенты\n" + "\n".join(ing_lines))

    steps = get_steps(rec)
    parts.append("## Приготовление\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
    return "\n\n".join(parts)


def _join_ru(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]


def make_user_prompt(rec: dict, rng: random.Random, mix: dict[str, float]) -> tuple[str, str]:
    """Pick a prompt style for a record. Returns (prompt_type, user_text)."""
    kinds, weights = zip(*mix.items())
    kind = rng.choices(kinds, weights=weights, k=1)[0]
    title = clean_text(rec["title"])

    if kind == "ingredients":
        names = [clean_text(i.get("name")).lower() for i in get_ingredients(rec)]
        names = [n for n in names if n and not any(s in n for s in _STAPLES)]
        if len(names) >= 2:
            k = min(len(names), rng.randint(2, 4))
            return kind, rng.choice(INGREDIENT_TEMPLATES).format(ings=_join_ru(rng.sample(names, k)))
        kind = "title"

    if kind == "category":
        crumbs = [clean_text(c) for c in as_list(rec.get("kroshki")) if clean_text(c)]
        if crumbs:
            extra = ""
            tastes = [clean_text(t).lower() for t in as_list(rec.get("tastes")) if clean_text(t)]
            purposes = [clean_text(p).lower() for p in as_list(rec.get("purposes")) if clean_text(p)]
            if tastes and rng.random() < 0.5:
                extra += ", вкус: " + ", ".join(tastes[:2])
            if purposes and rng.random() < 0.5:
                extra += ", повод: " + purposes[0]
            return kind, rng.choice(CATEGORY_TEMPLATES).format(cat=crumbs[-1], extra=extra)
        kind = "title"

    return "title", rng.choice(TITLE_TEMPLATES).format(title=title)


def is_usable(rec: dict, min_ingredients: int = 2, min_steps: int = 2, min_steps_chars: int = 80) -> bool:
    if not clean_text(rec.get("title")):
        return False
    ings = [i for i in get_ingredients(rec) if i and clean_text(i.get("name"))]
    steps = get_steps(rec)
    return len(ings) >= min_ingredients and len(steps) >= min_steps and sum(map(len, steps)) >= min_steps_chars


def ingredients_to_prompt(ingredients: Iterable[str]) -> str:
    items = [i.strip().lower() for i in ingredients if i.strip()]
    return INGREDIENT_TEMPLATES[0].format(ings=_join_ru(items))
