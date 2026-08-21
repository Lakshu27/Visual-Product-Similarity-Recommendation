"""
Step 1-2 of the pipeline: scan the image dataset, extract ResNet50 embeddings,
build a FAISS similarity index, and persist product metadata to SQLite.

Usage:
    python -m src.build_index --images_dir data/images --out_dir models
"""
import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

import config
from src.feature_extractor import ResNet50FeatureExtractor
from src.database import ProductDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def discover_images(images_dir: str) -> pd.DataFrame:
    """
    Build a metadata DataFrame from the images directory.

    Supports:
      - Category-subfolder layout: images_dir/<category>/<file>
      - Flat layout with an existing data/metadata.csv (loaded as-is)
    """
    if os.path.exists(config.METADATA_CSV):
        logger.info("Loading existing metadata from %s", config.METADATA_CSV)
        df = pd.read_csv(config.METADATA_CSV)
        required = {"image_path", "category"}
        if not required.issubset(df.columns):
            raise ValueError(f"metadata.csv must contain columns: {required}")
        return df

    logger.info("No metadata.csv found — auto-discovering from folder structure: %s", images_dir)
    records = []
    for root, _dirs, files in os.walk(images_dir):
        for fname in files:
            if fname.lower().endswith(config.IMAGE_EXTENSIONS):
                full_path = os.path.join(root, fname)
                category = os.path.basename(root) if root != images_dir else "uncategorized"
                product_id = os.path.splitext(fname)[0]
                records.append(
                    {
                        "image_path": full_path,
                        "category": category,
                        "product_id": product_id,
                        "price": None,
                    }
                )

    if not records:
        raise FileNotFoundError(
            f"No images found under '{images_dir}'. Add images following the "
            f"format described in README.md before running this script."
        )

    df = pd.DataFrame(records)
    df.to_csv(config.METADATA_CSV, index=False)
    logger.info("Auto-generated metadata.csv with %d entries.", len(df))
    return df


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an exact cosine-similarity FAISS index (IndexFlatIP over L2-normalized
    vectors). For datasets beyond ~1M images, swap this for IndexIVFFlat or
    IndexHNSWFlat for approximate, faster search.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def run(images_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = discover_images(images_dir)
    extractor = ResNet50FeatureExtractor()

    all_embeddings = []
    kept_rows = []

    paths = df["image_path"].tolist()
    logger.info("Extracting embeddings for %d images (batch_size=%d)...", len(paths), config.BATCH_SIZE)

    for start in tqdm(range(0, len(paths), config.BATCH_SIZE), desc="Embedding batches"):
        batch_paths = paths[start:start + config.BATCH_SIZE]
        embeddings, valid_paths = extractor.extract_batch(batch_paths)
        if embeddings is None:
            continue
        all_embeddings.append(embeddings)
        valid_set = set(valid_paths)
        batch_df = df.iloc[start:start + config.BATCH_SIZE]
        kept_rows.extend(
            [row for _, row in batch_df.iterrows() if row["image_path"] in valid_set]
        )

    if not all_embeddings:
        raise RuntimeError("No embeddings were extracted. Check that image paths in metadata.csv are valid.")

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")
    logger.info("Extracted embeddings matrix shape: %s", embeddings_matrix.shape)

    index = build_faiss_index(embeddings_matrix)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings_matrix)

    ids = [str(row.get("product_id", i)) for i, row in enumerate(kept_rows)]
    with open(os.path.join(out_dir, "ids.json"), "w") as f:
        json.dump(ids, f)

    db = ProductDatabase(os.path.join(out_dir, "products.db"))
    db.clear()
    db_rows = [
        (
            i,
            str(row.get("product_id", i)),
            row["image_path"],
            row.get("category"),
            row.get("price") if pd.notna(row.get("price")) else None,
        )
        for i, row in enumerate(kept_rows)
    ]
    db.bulk_insert(db_rows)

    logger.info(
        "Pipeline complete. Indexed %d products. Files written to '%s'.",
        db.count(),
        out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the visual similarity FAISS index.")
    parser.add_argument("--images_dir", default=config.IMAGES_DIR, help="Path to product images folder.")
    parser.add_argument("--out_dir", default=config.MODELS_DIR, help="Where to save index/embeddings/db.")
    args = parser.parse_args()

    try:
        run(args.images_dir, args.out_dir)
    except Exception as exc:
        logger.error("Build pipeline failed: %s", exc)
        raise
