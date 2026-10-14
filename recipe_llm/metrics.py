"""Automatic quality metrics for generated recipes (no reference answer needed).

* structure_ok         - it is a recipe: >=2 ingredients and >=2 numbered steps (any markdown style)
* house_format         - exactly the target layout: "# Title", "## Ингредиенты", "## Приготовление"
* must_have_recall     - classic dishes: share of key ingredients present (борщ -> свёкла, капуста)
* requested_coverage   - "У меня есть: X, Y": share of requested products used in the recipe
* ingredient_usage     - share of listed (non-staple) ingredients actually used in the steps
* quantities_share     - share of ingredient lines with an amount
* chatter              - forum-style phrases ("Приятного аппетита!!!", "у меня была...")
* repetition           - share of duplicated lines (degenerate loops)
* truncated            - generation hit max_new_tokens without finishing
"""
from __future__ import annotations

import re
from statistics import mean

from .textutil import is_staple, mentions, norm

_HEADING = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$")
# Section labels written without markdown headings: "**Ингредиенты:**", "Ингредиенты:"
_LABEL = re.compile(r"^\s*\**\s*([А-ЯЁа-яё ]{4,40}?)\s*:?\s*\**\s*:?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s+(.+?)[;.]?\s*$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.+)$")
_QTY = re.compile(r"\d|по вкусу|щепот|пучок|горст|половин|четверт", re.IGNORECASE)
_CHATTER = re.compile(r"приятного аппетита|у меня (?:был|была|было|были)\b|очень вкусн|попробуйте|!!|"
                      r"всем советую|мои домашние|рекомендую", re.IGNORECASE)
_HOUSE = re.compile(r"^#\s+\S.*\n(?:.*\n)*?##\s+Ингредиенты\s*\n(?:.*\n)*?##\s+Приготовление", re.MULTILINE)


def _section(name: str) -> str | None:
    if "ингредиент" in name or "понадоб" in name or "продукт" in name:
        return "ing"
    if "приготов" in name or "способ" in name or "шаг" in name or "инструкц" in name:
        return "steps"
    return None


def parse_recipe(text: str) -> dict:
    """Parses both the target layout and free-form markdown (###, *, bold labels)."""
    title, ingredients, steps, section = "", [], [], None
    for line in (text or "").splitlines():
        h = _HEADING.match(line) or _LABEL.match(line)
        if h:
            name = norm(h.group(1))
            sec = _section(name)
            if sec:
                section = sec
            elif not title and not ingredients and not steps and _HEADING.match(line):
                title, section = re.sub(r"^рецепт\s*:?\s*", "", h.group(1), flags=re.IGNORECASE), None
            elif _HEADING.match(line):
                section = None
            continue
        if section == "ing":
            m = _BULLET.match(line)
            if m:
                ingredients.append(m.group(1).strip())
        elif section == "steps":
            m = _NUMBERED.match(line)
            if m:
                steps.append(m.group(1).strip())
            elif steps and line.strip():
                steps[-1] += " " + line.strip()
    if not steps:  # no headings: fall back to any numbered lines
        steps = [m.group(1) for m in map(_NUMBERED.match, (text or "").splitlines()) if m]
    return {"title": title, "ingredients": ingredients, "steps": steps}


def ingredient_name(line: str) -> str:
    return re.split(r"\s+[—–-]\s+|:\s|\s\d", line, maxsplit=1)[0].strip()


def recipe_metrics(text: str, must_have: list[str] | None = None, requested: list[str] | None = None,
                   truncated: bool | None = None) -> dict:
    r = parse_recipe(text)
    ings, steps = r["ingredients"], r["steps"]
    body = norm("\n".join(ings + steps))
    steps_text = " ".join(steps)
    lines = [norm(l.strip()) for l in ings + steps if l.strip()]
    named = [ingredient_name(l) for l in ings]
    named = [n for n in named if n and not is_staple(n)]

    m = {
        "structure_ok": float(len(ings) >= 2 and len(steps) >= 2),
        "house_format": float(bool(_HOUSE.search((text or "").strip() + "\n"))),
        "chatter": float(bool(_CHATTER.search(text or ""))),
        "n_ingredients": len(ings),
        "n_steps": len(steps),
        "chars": len(text or ""),
        "quantities_share": mean([bool(_QTY.search(l)) for l in ings]) if ings else 0.0,
        "ingredient_usage": mean([mentions(steps_text, n) for n in named]) if named else 0.0,
        "repetition": 1 - len(set(lines)) / len(lines) if lines else 0.0,
    }
    if must_have:
        m["must_have_recall"] = mean([bool(re.search(p, body)) for p in must_have])
    if requested:
        m["requested_coverage"] = mean([mentions(body, q) for q in requested if q.strip()])
    if truncated is not None:
        m["truncated"] = float(truncated)
    return m


def summarize(rows: list[dict]) -> dict:
    """Mean of every metric over the rows that have it; also per prompt type."""
    out = {"n": len(rows)}
    keys = sorted({k for r in rows for k in r["metrics"]})
    for k in keys:
        vals = [r["metrics"][k] for r in rows if k in r["metrics"]]
        out[k] = round(mean(vals), 4)
    for t in sorted({r.get("type", "all") for r in rows}):
        sub = [r for r in rows if r.get("type") == t]
        for k in ("structure_ok", "must_have_recall", "requested_coverage", "ingredient_usage"):
            vals = [r["metrics"][k] for r in sub if k in r["metrics"]]
            if vals:
                out[f"{t}/{k}"] = round(mean(vals), 4)
    return out
