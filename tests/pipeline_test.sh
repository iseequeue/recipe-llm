#!/usr/bin/env bash
# Full pipeline on CPU with a tiny random YandexGPT-like model (Llama arch, " Ассистент:[SEP]" template,
# no system role, [NL] newlines) and a tiny sentence encoder. No downloads, ~3-4 min.
# data cleaning -> LLM judge -> LoRA training -> evaluation of base & fine-tuned -> RAG index ->
# evaluation with RAG -> SBS page -> RESULTS.md -> CLI and app (with RAG).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK=${WORK:-/tmp/recipe_pipeline}
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$ROOT" "$WORK/proj"; cd "$WORK/proj"
rm -rf data outputs eval/results eval/sbs_* eval/figures eval/REPORT.md eval/report_stats.json eval/judge_* mlflow.db mlruns
export MLFLOW_TRACKING_URI="sqlite:///$WORK/proj/mlflow.db" MLFLOW_DISABLE_AGENT_HINT=1 HF_HUB_OFFLINE=1
M="$WORK/assets/tiny-yandexgpt"
ENC="$WORK/assets/tiny-e5-encoder"

python tests/make_tiny_assets.py "$WORK/assets"

python scripts/build_dataset.py --recipes "$WORK/assets/recipes.parquet" --likes "$WORK/assets/likes/**/*.parquet" \
  --min_steps_chars 60 --min_avg_step_chars 10 --mlflow
python scripts/score_recipes.py --model "$M" --chunk_size 100 --top_k 250 --min_score 1 --device cpu --mlflow

python train.py --config configs/yandexgpt5-lite-8b_v100-32gb.yaml model_name="$M" use_cpu=true \
  max_train_samples=200 max_eval_samples=16 max_steps=12 per_device_train_batch_size=4 \
  gradient_accumulation_steps=1 eval_steps=6 save_steps=6 logging_steps=3 sample_max_new_tokens=24 \
  learning_rate=1e-3 preprocessing_num_workers=1 dataloader_num_workers=0

A=outputs/yandexgpt5-recipes/final_adapter
grep -q "newline_token: auto" "$A/recipe_training_config.yaml"
python scripts/evaluate.py --name base --base_model "$M" --max_new_tokens 40 --device cpu
python scripts/evaluate.py --name ft --adapter "$A" --max_new_tokens 40 --device cpu
python scripts/evaluate.py --name ft --rescore --no_mlflow
# --resume: keep the first 60 answers, generate only the rest
python -c "import json; L=open('eval/results/base.jsonl').readlines(); open('eval/results/base.jsonl','w').writelines(L[:60]); json.dump([json.loads(l)['response'] for l in L[:60]], open('$WORK/base60.json','w'))"
python scripts/evaluate.py --name base --base_model "$M" --max_new_tokens 40 --device cpu --resume --no_mlflow > "$WORK/resume.log"
grep "resume: 60 of" "$WORK/resume.log"
python -c "
import json; L=[json.loads(l) for l in open('eval/results/base.jsonl')]; P=[json.loads(l) for l in open('eval/prompts.jsonl')]
assert [r['id'] for r in L]==[p['id'] for p in P] and [r['response'] for r in L[:60]]==json.load(open('$WORK/base60.json')); print('resume ok', len(L))"
python infer.py --adapter "$A" --ingredients "творог, яйца" --max_new_tokens 20 --device cpu

# RAG: index over the recipes with a tiny encoder, then generation and evaluation with retrieval
python scripts/build_rag_index.py --recipes "$WORK/assets/recipes.parquet" --encoder "$ENC" --device cpu \
  --filter_args="--min_steps_chars 60 --min_avg_step_chars 10"
python scripts/evaluate.py --name ft-rag --adapter "$A" --rag data/rag_index --max_new_tokens 40 --device cpu
python scripts/evaluate.py --name ft-rag --rescore --no_mlflow
python -c "import json; s=json.load(open('eval/results/ft-rag.summary.json')); assert s['rag'] and 'retrieval_quality' in s, s"
python infer.py --adapter "$A" --rag data/rag_index --prompt "Напиши рецепт: Борщ" --max_new_tokens 20 --device cpu

python scripts/make_sbs.py base ft
python scripts/make_sbs.py ft ft-rag
python scripts/compare.py base ft ft-rag
python scripts/make_judge_labeling.py --n 30
python - <<'LBLEOF'
import json, re
html = open("eval/judge_labeling.html", encoding="utf-8").read()
items = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1).replace("<\\/", "</"))["items"]
assert len(items) == 30 and all(it["text"] for it in items)
json.dump({"items": [{"recipe_id": it["id"], "human": 1 + k % 5} for k, it in enumerate(items)]},
          open("eval/judge_labels.json", "w"))
LBLEOF
python scripts/report.py --bootstrap 500 --runs "tiny=%"
test -s eval/REPORT.md && test -s eval/figures/metrics_ci.png && test -s eval/figures/dishes_heatmap.png
test -s eval/figures/judge_validation.png && grep -q "Проверка судьи" eval/REPORT.md
python scripts/make_gallery.py --out "$WORK/gallery.html" && grep -q '"systems"' "$WORK/gallery.html"

A="$A" python - <<'PYEOF'
import os, sys
sys.argv = ["app.py", "--adapter", os.environ["A"], "--device", "cpu", "--rag", "data/rag_index", "--build_only"]
import app
demo, run = app.main()
outs = list(run("Блюдо", "Борщ", 0.0, 16, True, True))
assert isinstance(outs[-1][0], str) and outs[-1][1] and "Найдено" in outs[-1][2], outs[-1]  # tiny ft model may emit EOS at once
print("app with RAG ok:", outs[-1][2][:120])
PYEOF

python - <<'PYEOF'
import mlflow
names = sorted(e.name for e in mlflow.search_experiments())
assert {"recipe-data", "recipe-llm", "recipe-eval"} <= set(names), names
print("MLflow experiments:", names)
PYEOF
echo; echo "PIPELINE TEST PASSED"