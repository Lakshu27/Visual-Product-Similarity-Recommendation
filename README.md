# Visual Product Similarity & Image-Based Recommendation System (Amazon-Style)

Computer-vision based product recommendation engine. Given a product image, it returns
the Top-K visually similar products using deep learning embeddings (ResNet50) and
fast approximate nearest neighbor search (FAISS) — similar to Amazon's "Similar Items"
and image search features.

## Problem Statement
Online marketplaces host millions of products where textual metadata (title,
description, tags) is often noisy, incomplete, or misleading. Traditional
keyword-based search fails when users want visually similar products (same
style, color, or design). This project builds a computer vision–based system
that identifies and recommends visually similar products using only product
images.

## Business Use Case
- Enable image-based product search (upload an image → find similar items)
- Improve product discovery when text search fails
- Increase conversion rate by recommending visually relevant products
- Reduce dependency on manually curated product metadata
- Improve user experience for fashion, furniture, accessories, and home décor

## Project Objective
- Extract high-quality visual embeddings from product images using a pretrained CNN.
- Build a fast similarity search engine (FAISS) to retrieve visually similar products.
- Rank and return Top-K similar products for any input image.
- Provide a Streamlit UI for interactive image-based search.
- Demonstrate a scalable, production-like recommendation pipeline.

## Tech Stack
- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Model:** ResNet50 (pretrained on ImageNet, classification head removed)
- **Similarity Search:** FAISS (cosine similarity via normalized inner product)
- **Distance Metric:** Cosine Similarity
- **Database:** SQLite (product metadata — id, image path, category, price)
- **Frontend:** Streamlit

## Approach
**Step 1 — Image Feature Extraction:** Use a pretrained ResNet50, remove the
final classification layer, convert each product image into a dense 2048-d
embedding vector.

**Step 2 — Embedding Indexing:** Store all image embeddings and build a FAISS
similarity index for millisecond-level search.

**Step 3 — Similarity Matching:** Given a query image, extract its embedding,
perform cosine similarity search, retrieve Top-K visually similar products.

**Step 4 — Ranking & Filtering:** Rank results by similarity score; optionally
filter by category.

**Step 5 — Evaluation:** Precision@K, Recall@K, and visual inspection.

## Folder Structure
```
visual-product-similarity/
├── data/
│   ├── images/            <- product images, organized by category
│   └── metadata.csv       <- auto-generated from folder structure
├── models/                <- generated: embeddings.npy, faiss.index, ids.json, products.db
├── raw_dataset/            <- raw Kaggle download (images/ + styles.csv)
├── src/
│   ├── database.py         SQLite metadata store
│   ├── feature_extractor.py ResNet50 embedding extractor
│   ├── build_index.py      Pipeline: scan images -> embeddings -> FAISS index -> DB
│   ├── search_engine.py    Load index + search Top-K for a query image
│   └── evaluate.py         Precision@K / Recall@K evaluation
├── app.py                  Streamlit app (image upload -> Top-K results)
├── tests/test_basic.py     Sanity tests
├── scripts/
│   ├── run_pipeline.sh          Builds index + runs evaluation
│   └── reorganize_dataset.py    Sorts Kaggle dataset into category folders
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install kaggle
```

## Dataset — Download & Prepare

This project uses the **Kaggle "Fashion Product Images (Small)"** dataset
(44k product images with category labels). Other public/industry-accepted
options for this project type include the Stanford Online Products Dataset
and the DeepFashion Dataset — any one dataset is sufficient for a strong
portfolio project. Recommended minimum size: 1000 images.

### 1. Get a Kaggle API key
1. Go to [kaggle.com](https://www.kaggle.com) → sign in → click your profile
   picture → **Settings** → **API Tokens** tab.
2. Under **Legacy API Credentials**, click **Create Legacy API Key** — this
   downloads a `kaggle.json` file (contains your username + key).
3. Move it into place:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 2. Download the dataset
```bash
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d raw_dataset
```
This creates `raw_dataset/images/` (product photos) and
`raw_dataset/styles.csv` (category labels, keyed by product `id`).

### 3. Reorganize into category folders
`build_index.py` expects images sorted as `data/images/<category>/*.jpg`.
Run the reorganizer to sample a balanced subset:
```bash
python scripts/reorganize_dataset.py
```
By default this copies up to 40 images per category (skipping categories
with fewer than 15 images), capped at 2000 images total — enough for a
strong portfolio demo while keeping embedding extraction fast on CPU.
Tune it if you want more/fewer images:
```bash
python scripts/reorganize_dataset.py --max_per_category 60 --max_total 4000
```

**Alternative dataset layout (flat folder + your own metadata.csv):**
```
data/images/*.jpg
data/metadata.csv   columns: image_path,category,price,product_id
```

## Usage

### 1. Build the index
```bash
bash scripts/run_pipeline.sh
```
This will:
- Extract a ResNet50 embedding for every image in `data/images/`
- Build a FAISS cosine-similarity index
- Store metadata in a SQLite DB (`models/products.db`)
- Print Precision@K / Recall@K evaluation results

Or run the steps individually:
```bash
python -m src.build_index --images_dir data/images --out_dir models
python -m src.evaluate --top_k 5
```

### 2. Search similar products from the command line
```bash
python -m src.search_engine --query data/images/Jackets/13240.jpg --top_k 5
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` (or `8502`/`8503` if already in use).
Upload a **product image** (use one from `data/images/<category>/` for a
realistic test) to see Top-K visually similar results with similarity
scores, ranked and filterable by category.

## Results & Evaluation

**Quantitative results (actual, on this dataset — 2000 products, 50 categories, Top-5):**
- **Precision@5: 0.6676** — ~3.3 of every 5 retrieved results share the
  query's category. High Precision@K, matching the project's target.
- **Recall@5: 0.0856** — expected to be low at small K since each category
  has ~40 images (max possible recall@5 ≈ 5/39 ≈ 0.128); recall rises with
  larger K.
- Sub-second similarity search using FAISS.
- Robust recommendations even when text metadata is missing.

To re-run evaluation with a different K:
```bash
python -m src.evaluate --top_k 10
```

**Qualitative results:** Visually coherent recommendations (same style,
color, structure) — confirmed via the Streamlit app using real product
images as queries. Note: querying with a non-product image (e.g. a person's
photo) will naturally produce low-similarity, unrelated results — ResNet50
embeddings are meaningful for product photos, not arbitrary images.

## Notes / Design Decisions
- Embeddings are extracted from ResNet50's penultimate (2048-d) layer with
  the final FC classification layer removed, then L2-normalized so that
  FAISS inner-product search is equivalent to cosine similarity.
- FAISS `IndexFlatIP` is used (exact search, ideal for portfolio-scale
  datasets). Swap to `IndexIVFFlat`/`IndexHNSWFlat` for datasets beyond
  ~1M vectors — see comments in `src/build_index.py`.
- The Streamlit app loads the index once (`st.cache_resource`) and paginates
  gallery results rather than loading the entire dataset into memory.
- All scripts are modular, PEP8-formatted, documented with docstrings, and
  wrap I/O and model calls in try/except with clear error messages.
- SQLite metadata table is normalized (single table, indexed on `category`)
  for fast filtering.

## Testing
```bash
python -m pytest tests/test_basic.py -v
```
Covers database CRUD, FAISS index roundtrip, and config sanity.

## Re-running from scratch
If you want to rebuild with a different sample of images:
```bash
rm -rf data/images models/*
python scripts/reorganize_dataset.py --max_per_category 50
bash scripts/run_pipeline.sh
```