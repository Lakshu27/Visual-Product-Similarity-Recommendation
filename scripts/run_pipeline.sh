#!/usr/bin/env bash
# One-shot: build the FAISS index from data/images, then print evaluation metrics.
set -e

cd "$(dirname "$0")/.."

echo "==> Building embeddings + FAISS index..."
python -m src.build_index --images_dir data/images --out_dir models

echo ""
echo "==> Evaluating Precision@K / Recall@K..."
python -m src.evaluate --top_k 5

echo ""
echo "Done. Launch the UI with: streamlit run app.py"
