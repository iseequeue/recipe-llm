"""Asserts the last MLflow run has the expected metrics/params/artifacts."""
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

uri = sys.argv[1]
mlflow.set_tracking_uri(uri)
client = MlflowClient()
exp = client.get_experiment_by_name("recipe-llm")
run = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)[0]
m = run.data.metrics
for key in ["loss", "eval_loss", "eval_perplexity", "data/train_examples", "final/eval_loss", "learning_rate"]:
    assert key in m, f"missing metric {key}; have {sorted(m)}"
assert "cfg.model_name" in run.data.params

def walk(path=""):
    for a in client.list_artifacts(run.info.run_id, path):
        yield from (walk(a.path) if a.is_dir else [a.path])

arts = set(walk())
for a in ["config.yaml", "data_preview.md", "system_prompt.txt", "samples/final.md",
          "adapter/adapter_model.safetensors", "adapter/adapter_config.json", "adapter/tokenizer_config.json"]:
    assert a in arts, f"missing artifact {a}; have {sorted(arts)}"
assert any(a.startswith("samples/step_") for a in arts)
print(f"MLflow OK: run {run.info.run_id}, {len(m)} metrics, {len(arts)} artifacts")
print("  metric history (eval_loss):", [round(x.value, 3) for x in client.get_metric_history(run.info.run_id, "eval_loss")])
with open(os.path.join(os.path.dirname(uri.replace("sqlite:///", "")), "run_id"), "w") as f:
    f.write(run.info.run_id)
