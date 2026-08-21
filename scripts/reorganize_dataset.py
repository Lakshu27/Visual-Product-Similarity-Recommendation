"""
Reorganize the Kaggle "Fashion Product Images (Small)" dataset into the
category-folder layout expected by this project's build_index.py:

    data/images/<articleType>/<id>.jpg

Usage:
    python scripts/reorganize_dataset.py \
        --raw_dir raw_dataset \
        --out_dir data/images \
        --max_per_category 40 \
        --min_per_category 15

By default this samples a manageable, class-balanced subset (rather than all
44k images) so ResNet50 embedding extraction finishes in a reasonable time on
a CPU. Categories (articleType values) with fewer than --min_per_category
images are skipped, since they can't contribute meaningful Precision@K /
Recall@K signal.
"""
import os
import csv
import shutil
import argparse
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_rows(styles_csv):
    """Read styles.csv robustly (some rows have extra commas in the display name)."""
    rows = []
    with open(styles_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id") and row.get("articleType"):
                rows.append(row)
    return rows


def run(raw_dir, out_dir, max_per_category, min_per_category, max_total):
    images_dir = os.path.join(raw_dir, "images")
    styles_csv = os.path.join(raw_dir, "styles.csv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Could not find images folder at '{images_dir}'")
    if not os.path.isfile(styles_csv):
        raise FileNotFoundError(f"Could not find styles.csv at '{styles_csv}'")

    rows = load_rows(styles_csv)
    logger.info("Loaded %d rows from styles.csv", len(rows))

    by_category = defaultdict(list)
    for row in rows:
        image_path = os.path.join(images_dir, f"{row['id']}.jpg")
        if os.path.exists(image_path):
            by_category[row["articleType"]].append((row["id"], image_path))

    os.makedirs(out_dir, exist_ok=True)

    total_copied = 0
    kept_categories = 0

    for category, items in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        if len(items) < min_per_category:
            continue
        if max_total and total_copied >= max_total:
            break

        sample = items[:max_per_category]
        cat_dir = os.path.join(out_dir, category.replace("/", "-").strip())
        os.makedirs(cat_dir, exist_ok=True)

        for product_id, src_path in sample:
            dst_path = os.path.join(cat_dir, f"{product_id}.jpg")
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            total_copied += 1

        kept_categories += 1
        logger.info("Category '%s': copied %d images", category, len(sample))

    logger.info(
        "Done. %d images copied across %d categories into '%s'.",
        total_copied, kept_categories, out_dir,
    )
    if total_copied == 0:
        logger.warning("No images were copied — check --min_per_category / paths.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize Kaggle fashion dataset into category folders.")
    parser.add_argument("--raw_dir", default="raw_dataset", help="Path to the unzipped Kaggle dataset.")
    parser.add_argument("--out_dir", default="data/images", help="Destination for category-organized images.")
    parser.add_argument("--max_per_category", type=int, default=40, help="Cap images per category.")
    parser.add_argument("--min_per_category", type=int, default=15, help="Skip categories with fewer images than this.")
    parser.add_argument("--max_total", type=int, default=2000, help="Overall cap on copied images (0 = no cap).")
    args = parser.parse_args()

    try:
        run(
            args.raw_dir,
            args.out_dir,
            args.max_per_category,
            args.min_per_category,
            args.max_total if args.max_total > 0 else None,
        )
    except Exception as exc:
        logger.error("Reorganization failed: %s", exc)
        raise
