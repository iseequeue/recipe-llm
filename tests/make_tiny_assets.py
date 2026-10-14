"""Builds offline test assets: synthetic recipes in the povarenok schema and a
tiny randomly initialised Qwen2 model with a BPE tokenizer and ChatML template.
Lets the whole train/infer pipeline run on CPU without downloading anything.

python tests/make_tiny_assets.py tests/assets
"""
import os
import random
import sys

import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (BertConfig, BertModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          Qwen2Config, Qwen2ForCausalLM)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from recipe_llm.prompts import SYSTEM_PROMPT, format_recipe  # noqa: E402

# Mimics YandexGPT-5-Lite: role prefixes, no system role, generation prompt " Ассистент:[SEP]".
YANDEX_LIKE = (
    "{{ bos_token }}{%- set names = {'assistant': ' Ассистент:', 'user': ' Пользователь:'} %}"
    "{%- for message in messages %}{{ names[message['role']] }} {{ message['content'] }}"
    "{%- if message['role'] == 'assistant' %}{{ eos_token }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %} Ассистент:[SEP]{%- endif %}"
)

CHATML = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

DISHES = ["Борщ", "Сырники", "Плов", "Пирог с яблоками", "Салат Оливье", "Котлеты", "Блины", "Солянка"]
INGS = ["мука", "яйцо", "молоко", "сахар", "соль", "картофель", "морковь", "лук", "говядина", "творог",
        "яблоки", "рис", "свёкла", "капуста", "сметана", "масло сливочное"]
CATS = [["Супы", "Горячие супы"], ["Выпечка", "Выпечка сладкая"], ["Вторые блюда"], ["Салаты"]]
STEPS = ["Нарезать {a} кубиками.", "Смешать {a} и {b} до однородности.", "Обжарить {a} на среднем огне 5 минут.",
         "Добавить {a} и тушить 15 минут под крышкой.", "Выпекать при 180 градусах 30 минут.",
         "Подавать горячим со &quot;сметаной&quot;."]


def make_records(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        ings = rng.sample(INGS, rng.randint(3, 7))
        rows.append({
            "title": f"{rng.choice(DISHES)} №{i}",
            "page_url": f"https://example.org/{i}",
            "description": "Простое домашнее блюдо.\r\n\r\n\r\nОчень вкусно!" if i % 3 else None,
            "views": rng.randint(10, 100000),
            "kroshki": rng.choice(CATS),
            "ingredients": [{"count": rng.choice(["200 г", "2 шт", "1 ст. л.", None]), "name": x.capitalize()} for x in ings],
            "cooking_time": rng.choice(["30 минут", "1 час", None]),
            "portions_count": rng.choice(["2", "4", None]),
            "cooking_steps": [rng.choice(STEPS).format(a=rng.choice(ings), b=rng.choice(ings)) for _ in range(rng.randint(3, 6))] + [""],
            "purposes": rng.choice([[], ["На праздничный стол"], ["Для детей"]]),
            "tastes": rng.choice([[], ["сладкий"], ["солёный", "пряный"]]),
            "main_image_url": "x.jpg",
            "page_url": f"https://www.povarenok.ru/recipes/show/{1000 + i}/",
        })
    rows[0]["ingredients"] = rows[0]["ingredients"][:1]  # must be filtered out (1 ingredient)
    return rows


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = make_records(400)
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "recipes.parquet"))

    corpus = [SYSTEM_PROMPT] + [format_recipe(r) for r in rows] * 2 + ["Напиши рецепт: Предложи раздел вкус повод"]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=specials[1:])
    tok.chat_template = CHATML
    model_dir = os.path.join(out_dir, "tiny-qwen")
    tok.save_pretrained(model_dir)

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(model_dir)

    # Yandex-like: Llama architecture, <s>/</s>, [SEP], newlines as [NL] (dir name contains "yandexgpt").
    ycorpus = [c.replace("\n", "[NL]") for c in corpus] + ["Пользователь: Ассистент:[SEP] Рецепт: 1 2 3 4 5"]
    yspecials = ["<unk>", "<s>", "</s>", "[SEP]"]
    ytk = Tokenizer(models.BPE(unk_token="<unk>"))
    ytk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    ytk.decoder = decoders.ByteLevel()
    ytk.train_from_iterator(ycorpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=yspecials,
                                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    ytok = PreTrainedTokenizerFast(tokenizer_object=ytk, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    ytok.chat_template = YANDEX_LIKE
    ydir = os.path.join(out_dir, "tiny-yandexgpt")
    ytok.save_pretrained(ydir)
    ycfg = LlamaConfig(vocab_size=len(ytok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                       bos_token_id=ytok.bos_token_id, eos_token_id=ytok.eos_token_id, tie_word_embeddings=False)
    LlamaForCausalLM(ycfg).save_pretrained(ydir)

    # Tiny sentence encoder for RAG (name contains "e5" -> "query:"/"passage:" prefixes, mean pooling).
    edir = os.path.join(out_dir, "tiny-e5-encoder")
    tok.save_pretrained(edir)
    BertModel(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512,
                         pad_token_id=tok.pad_token_id)).save_pretrained(edir)

    # Second dump with likes (like dim/povarenok), covering half of the recipes.
    likes = [{"link": f"https://www.povarenok.ru/recipes/show/{1000 + i}/", "likes": random.Random(i).randint(1, 900),
              "ups": random.Random(-i).randint(60, 500), "views": rows[i]["views"]} for i in range(0, len(rows), 2)]
    os.makedirs(os.path.join(out_dir, "likes", "data"), exist_ok=True)
    pd.DataFrame(likes).to_parquet(os.path.join(out_dir, "likes", "data", "train-00000.parquet"))
    print(f"assets in {out_dir}: vocab={len(tok)}, yandex-like vocab={len(ytok)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/assets")
