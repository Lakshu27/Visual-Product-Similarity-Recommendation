"""
Step 5: Evaluation — Precision@K and Recall@K.

Ground truth: an image's "relevant" set is every other image sharing its
category. For each product in the index, we query with its own embedding
(excluding itself from results) and measure how many of the Top-K retrieved
items share its category.
"""
import argparse
import logging
from collections import Counter

import faiss
import numpy as np

import config
from src.database import ProductDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_artifacts():
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    embeddings = np.load(config.EMBEDDINGS_PATH)
    db = ProductDatabase(config.DB_PATH)
    return index, embeddings, db


def evaluate(top_k: int = config.DEFAULT_TOP_K, sample_size: int = None):
    index, embeddings, db = load_artifacts()
    n = embeddings.shape[0]

    rows = db.get_by_vector_indices(list(range(n)))
    category_by_index = {r[0]: r[3] for r in rows}
    category_counts = Counter(category_by_index.values())

    sample_indices = list(range(n))
    if sample_size and sample_size < n:
        rng = np.random.default_rng(42)
        sample_indices = sorted(rng.choice(n, size=sample_size, replace=False).tolist())

    precisions, recalls = [], []

    for vec_idx in sample_indices:
        query_category = category_by_index.get(vec_idx)
        if query_category is None:
            continue

        total_relevant = category_counts[query_category] - 1  # exclude the query itself
        if total_relevant <= 0:
            continue  # no possible positive matches for this category

        query_vector = embeddings[vec_idx].reshape(1, -1).astype("float32")
        scores, indices = index.search(query_vector, top_k + 1)  # +1 to allow removing self
        retrieved = [int(i) for i in indices[0] if int(i) != vec_idx][:top_k]

        relevant_retrieved = sum(
            1 for r in retrieved if category_by_index.get(r) == query_category
        )

        precisions.append(relevant_retrieved / top_k)
        recalls.append(relevant_retrieved / total_relevant)

    if not precisions:
        logger.warning("No evaluable samples (dataset may have only one image per category).")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "n_evaluated": 0}

    result = {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "n_evaluated": len(precisions),
        "top_k": top_k,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Precision@K / Recall@K.")
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--sample_size", type=int, default=None, help="Evaluate on a random subset for speed.")
    args = parser.parse_args()

    try:
        metrics = evaluate(top_k=args.top_k, sample_size=args.sample_size)
        print(f"Evaluated on {metrics['n_evaluated']} products (Top-{args.top_k})")
        print(f"Precision@{args.top_k}: {metrics['precision_at_k']:.4f}")
        print(f"Recall@{args.top_k}:    {metrics['recall_at_k']:.4f}")
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise
