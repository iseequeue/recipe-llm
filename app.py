#!/usr/bin/env python
"""Web demo: recipe generation with the fine-tuned model, optionally side by side with the base model.

  python app.py --adapter outputs/yandexgpt5-recipes/final_adapter --share   # public *.gradio.live link
  python app.py --adapter outputs/yandexgpt5-recipes/final_adapter --port 7860
  python app.py --adapter outputs/yandexgpt5-recipes/final_adapter --rag data/rag_index --share   # + search

The LoRA adapter is kept unmerged, so the same weights in GPU memory also give the base
model's answer (adapter switched off) for a "before / after" comparison.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipe_llm.generation import GenParams, load_model, stream  # noqa: E402
from recipe_llm.prompts import ingredients_to_prompt  # noqa: E402
from recipe_llm.rag import load_retriever  # noqa: E402

MODES = ["Блюдо", "Из продуктов", "Категория", "Свободный запрос"]
EXAMPLES = [
    ["Блюдо", "Борщ"], ["Блюдо", "Сырники"], ["Из продуктов", "курица, рис, морковь"],
    ["Из продуктов", "кабачки, фарш, сыр"], ["Категория", "Выпечка сладкая"],
    ["Свободный запрос", "Что приготовить на ужин за 30 минут без мяса?"],
]


def build_prompt(mode: str, text: str) -> str:
    text = (text or "").strip()
    if mode == "Блюдо":
        return f"Напиши рецепт: {text}"
    if mode == "Из продуктов":
        return ingredients_to_prompt(text.replace(";", ",").split(","))
    if mode == "Категория":
        return f"Предложи рецепт из раздела «{text}»."
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter")
    ap.add_argument("--base_model")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--rag", help="RAG index dir: adds a 'search similar recipes' switch")
    ap.add_argument("--rag_k", type=int, default=3)
    ap.add_argument("--build_only", action="store_true", help="build the UI and exit (tests)")
    args = ap.parse_args()

    import gradio as gr

    model, _, fmt = load_model(args.base_model, args.adapter, args.dtype, args.device, merge=False)
    has_adapter = hasattr(model, "disable_adapter")
    retriever = load_retriever(args.rag, args.device)

    def run(mode, text, temperature, max_tokens, compare, use_rag=False):
        if not (text or "").strip():
            yield "Введите запрос.", "", ""
            return
        prompt = build_prompt(mode, text)
        sources = ""
        if retriever and use_rag:
            prompt, hits = retriever.augment(prompt, args.rag_k)
            sources = "**Найдено в базе рецептов:** " + "; ".join(
                f"{h['title']} ({h['similarity']:.2f})" for h in hits)
        params = GenParams(max_new_tokens=int(max_tokens), temperature=float(temperature))
        ft = ""
        for ft in stream(model, fmt, prompt, params):
            yield ft, "", sources
        if compare and has_adapter:
            for base in stream(model, fmt, prompt, params, use_adapter=False):
                yield ft, base, sources

    with gr.Blocks(title="Рецепты: YandexGPT + LoRA") as demo:
        gr.Markdown("## Генератор рецептов\nYandexGPT-5-Lite-8B, дообученная (LoRA) на отобранных рецептах povarenok.ru.")
        with gr.Row():
            mode = gr.Radio(MODES, value="Блюдо", label="Что ищем")
            text = gr.Textbox(label="Запрос", placeholder="Борщ / курица, рис, морковь / Выпечка сладкая", scale=3)
        with gr.Row():
            temperature = gr.Slider(0.0, 1.2, value=0.5, step=0.05, label="Температура (0 = детерминированно)")
            max_tokens = gr.Slider(128, 1024, value=768, step=64, label="Макс. длина ответа, токенов")
            compare = gr.Checkbox(value=False, label="Показать ответ базовой модели рядом", visible=has_adapter)
            use_rag = gr.Checkbox(value=retriever is not None, label="Искать похожие рецепты в базе (RAG)",
                                  visible=retriever is not None)
        btn = gr.Button("Сгенерировать рецепт", variant="primary")
        sources = gr.Markdown()
        with gr.Row():
            out_ft = gr.Markdown(label="Дообученная модель")
            out_base = gr.Markdown(label="Базовая модель")
        gr.Examples(EXAMPLES, inputs=[mode, text])
        inputs = [mode, text, temperature, max_tokens, compare, use_rag]
        outputs = [out_ft, out_base, sources]
        btn.click(run, inputs, outputs)
        text.submit(run, inputs, outputs)

    if args.build_only:
        return demo, run
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
