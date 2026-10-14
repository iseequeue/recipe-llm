#!/usr/bin/env bash
# End-to-end smoke test on CPU with a tiny random model (no downloads, ~1-2 min).
# Checks: data pipeline -> LoRA training -> eval -> MLflow run/artifacts -> inference.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/tmp/recipe_smoke}
rm -rf "$WORK"; mkdir -p "$WORK"

python tests/make_tiny_assets.py "$WORK/assets"

python train.py --config configs/qwen2.5-1.5b_v100.yaml \
  model_name="$WORK/assets/tiny-qwen" data_files="$WORK/assets/recipes.parquet" \
  output_dir="$WORK/out" mlflow_tracking_uri="sqlite:///$WORK/mlflow.db" run_name=smoke \
  max_train_samples=120 max_eval_samples=16 max_seq_len=512 preprocessing_num_workers=1 \
  max_steps=8 per_device_train_batch_size=2 gradient_accumulation_steps=2 \
  eval_steps=4 save_steps=4 logging_steps=2 dataloader_num_workers=0 sample_max_new_tokens=24 \
  learning_rate=1e-3 use_cpu=true

python tests/check_mlflow.py "sqlite:///$WORK/mlflow.db"

python infer.py --adapter "$WORK/out/final_adapter" --prompt "Напиши рецепт: Сырники" \
  --max_new_tokens 30 --device cpu --seed 0
printf 'Напиши рецепт: Борщ\nПредложи рецепт из раздела «Салаты».\nДай рецепт блюда «Плов».\n' > "$WORK/prompts.txt"
python infer.py --adapter "$WORK/out/final_adapter" --input_file "$WORK/prompts.txt" \
  --output_file "$WORK/out.jsonl" --batch_size 2 --max_new_tokens 20 --device cpu --temperature 0
test "$(wc -l < "$WORK/out.jsonl")" -eq 3
echo "из: творог, яйцо" | python infer.py --adapter "$WORK/out/final_adapter" --interactive --max_new_tokens 20 --device cpu
MLFLOW_TRACKING_URI="sqlite:///$WORK/mlflow.db" python infer.py \
  --adapter "runs:/$(cat "$WORK/run_id")/adapter" --ingredients "рис, морковь" --max_new_tokens 20 --device cpu
python infer.py --adapter "$WORK/out/final_adapter" --merge_to "$WORK/merged" --device cpu
python infer.py --base_model "$WORK/merged" --prompt "Напиши рецепт: Блины" --max_new_tokens 20 --device cpu

echo; echo "SMOKE TEST PASSED"
