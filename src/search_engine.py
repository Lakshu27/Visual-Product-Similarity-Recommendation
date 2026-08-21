"""
Step 3-4 of the pipeline: given a query image, extract its embedding, search
the FAISS index for the Top-K nearest neighbors by cosine similarity, rank
them, and optionally filter by category/price.
"""
import os
import json
import argparse
import logging

import faiss

import config
from src.feature_extractor import ResNet50FeatureExtractor
from src.database import ProductDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SimilaritySearchEngine:
    """Loads a prebuilt FAISS index + metadata DB and serves Top-K queries."""

    def __init__(self, index_path=None, ids_path=None, db_path=None):
        index_path = index_path or config.FAISS_INDEX_PATH
        ids_path = ids_path or config.IDS_PATH
        db_path = db_path or config.DB_PATH

        for path in (index_path, ids_path, db_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing '{path}'. Run `python -m src.build_index` first."
                )

        self.index = faiss.read_index(index_path)
        with open(ids_path) as f:
            self.ids = json.load(f)
        self.db = ProductDatabase(db_path)
        self.extractor = ResNet50FeatureExtractor()

    def search(self, query_image_path: str, top_k: int = config.DEFAULT_TOP_K, category_filter: str = None):
        """
        Returns a list of dicts:
            {vector_index, product_id, image_path, category, price, similarity_score}
        ranked by descending cosine similarity.
        """
        query_embedding = self.extractor.extract_single(query_image_path)
        if query_embedding is None:
            raise ValueError(f"Could not extract features from query image: {query_image_path}")

        # Over-fetch when filtering by category so we still return top_k after filtering
        fetch_k = top_k * 5 if category_filter else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), fetch_k
        )
        scores, indices = scores[0], indices[0]

        valid_indices = [int(i) for i in indices if i != -1]
        rows = self.db.get_by_vector_indices(valid_indices)
        score_map = dict(zip(valid_indices, scores))

        results = []
        for row in rows:
            vector_index, product_id, image_path, category, price = row
            if category_filter and category != category_filter:
                continue
            results.append(
                {
                    "vector_index": vector_index,
                    "product_id": product_id,
                    "image_path": image_path,
                    "category": category,
                    "price": price,
                    "similarity_score": float(score_map.get(vector_index, 0.0)),
                }
            )
            if len(results) >= top_k:
                break

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search visually similar products for a query image.")
    parser.add_argument("--query", required=True, help="Path to the query image.")
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--category", default=None, help="Optional category filter.")
    args = parser.parse_args()

    try:
        engine = SimilaritySearchEngine()
        results = engine.search(args.query, top_k=args.top_k, category_filter=args.category)
        if not results:
            print("No results found.")
        for rank, r in enumerate(results, start=1):
            print(f"{rank}. {r['product_id']} | category={r['category']} | "
                  f"score={r['similarity_score']:.4f} | {r['image_path']}")
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise
