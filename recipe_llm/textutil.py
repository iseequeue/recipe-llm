"""Tiny Russian text helpers for data filtering and evaluation metrics (no external NLP deps)."""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_LETTERS_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
_TRAILING = "аеёиоуыэюяьй"
# Ingredients that are in almost every recipe or never named in steps - ignored in coverage metrics.
STAPLES = {"соль", "перец", "вода", "специи", "зелень", "сахар", "масло растительное"}


def norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def stem(word: str) -> str:
    """Crude stem: drop trailing vowels/ь/й, keep at most 6 letters.
    морковь->морков (matches 'морковку'), яйцо->яйц ('яйца'), свекла->свекл ('свеклу')."""
    w = norm(word)
    while len(w) > 3 and w[-1] in _TRAILING:
        w = w[:-1]
    return w[:6]


def head_word(name: str) -> str:
    words = _LETTERS_RE.findall(norm(name))
    return words[0] if words else ""


def ingredient_stem(name: str) -> str:
    return stem(head_word(name))


def is_staple(name: str) -> bool:
    n = norm(name).strip()
    return any(n.startswith(s) for s in STAPLES)


# Same product, different words: "картофель" in the list, "картошку" in the steps.
ALIASES = {  # keys are stems (<= 6 letters)
    "картоф": ["картош"], "яйц": ["яиц", "яйк", "желт", "белк"], "помидо": ["томат"], "томат": ["помидор"],
    "куриц": ["кур", "филе"], "курин": ["кур", "филе"], "фил": ["кур", "мяс"], "фарш": ["мяс"],
    "сливоч": ["масл"], "растит": ["масл"], "репчат": ["лук"], "овсянк": ["овсян"],
}


def _stem_in(st: str, t: str) -> bool:
    # Short stems are matched as a word start with a short ending: "рис" -> рис/риса/рисом but not "присыпать",
    # "нут" -> нут/нута but not "минут", "сод" -> сода/соду but not "содержимое".
    if len(st) <= 3:
        return re.search(rf"(?<![а-я]){st}[а-я]{{0,3}}(?![а-я])", t) is not None
    return st in t


def mentions(text: str, ingredient_name: str) -> bool:
    st = ingredient_stem(ingredient_name)
    t = norm(text)
    return bool(st) and (_stem_in(st, t) or any(a in t for a in ALIASES.get(st, [])))


def cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in letters) / len(letters)


def normalize_title(title: str) -> str:
    return " ".join(_WORD_RE.findall(norm(title)))
