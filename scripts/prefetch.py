#!/usr/bin/env python
"""Download model weights and the povarenok dataset so that training can run offline
(HPC compute nodes usually have no internet). Run on a machine with internet access.

python scripts/prefetch.py                                     # YandexGPT-5-Lite-8B-instruct + datasets
python scripts/prefetch.py --models yandex/YandexGPT-5-Lite-8B-instruct Qwen/Qwen2.5-7B-Instruct
python scripts/prefetch.py --skip_data --models --encoders intfloat/multilingual-e5-large   # for RAG
"""
from __future__ import annotations

import argparse
import glob
import os

# Same default cache as slurm/env.sh uses in jobs (set before importing huggingface_hub).
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_DIR, "hf_cache"))
# Plain HTTP download: the Xet backend keeps its own chunk cache next to the files (~2x disk for a moment).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# This script exists to download: ignore offline flags left in the shell by `source slurm/env.sh`.
for _v in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(_v, None)

from huggingface_hub import snapshot_download  # noqa: E402

DATASET_REPO = "d0rj/povarenok_recipes_detail"
LIKES_REPO = "dim/povarenok"  # second dump of the same site: has likes/ups, used as a quality signal
MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "original_tokenizer/*"]


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9


def fetch_dataset(data_dir: str) -> None:
    print(f"== dataset {DATASET_REPO} -> {data_dir}")
    snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=data_dir, allow_patterns=["data/*.parquet"])
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(data_dir, "data", "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files downloaded into {data_dir}/data")
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    cols = pq.ParquetFile(files[0]).schema_arrow.names
    missing = {"title", "ingredients", "cooking_steps"} - set(cols)
    if missing:
        raise SystemExit(f"Dataset schema changed, missing columns: {missing}")
    print(f"   {len(files)} files, {rows:,} recipes, {dir_size_gb(data_dir):.2f} GB")


def fetch_likes(data_dir: str) -> None:
    print(f"== likes {LIKES_REPO} -> {data_dir}")
    snapshot_download(LIKES_REPO, repo_type="dataset", local_dir=data_dir, allow_patterns=["*.parquet", "**/*.parquet"])
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    print(f"   {len(files)} files")


def fetch_model(repo: str, chat: bool = True) -> None:
    print(f"== model {repo} -> HF cache ({os.environ.get('HF_HOME', '~/.cache/huggingface')})")
    path = snapshot_download(repo, allow_patterns=MODEL_FILES)
    from transformers import AutoConfig, AutoTokenizer

    AutoConfig.from_pretrained(path)
    tok = AutoTokenizer.from_pretrained(path)
    if chat:
        assert tok.chat_template, f"{repo} has no chat template - pick an *-Instruct model"
    print(f"   ok, {dir_size_gb(path):.1f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["yandex/YandexGPT-5-Lite-8B-instruct"])
    ap.add_argument("--data_dir", default=os.path.join(PROJECT_DIR, "data", "povarenok"))
    ap.add_argument("--likes_dir", default=os.path.join(PROJECT_DIR, "data", "povarenok_likes"))
    ap.add_argument("--encoders", nargs="*", default=[], help="sentence encoders for RAG, e.g. intfloat/multilingual-e5-large")
    ap.add_argument("--skip_data", action="store_true")
    args = ap.parse_args()
    if not args.skip_data:
        fetch_dataset(args.data_dir)
        fetch_likes(args.likes_dir)
    for m in args.models:
        fetch_model(m)
    for m in args.encoders:
        fetch_model(m, chat=False)
    print("All downloads done. Jobs can now run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
