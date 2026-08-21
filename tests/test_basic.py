"""
Basic sanity tests. These check the pipeline's building blocks without
requiring a full dataset (heavier integration tests should be run manually
after `build_index.py` with real data).

Run with:
    python -m pytest tests/test_basic.py -v
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.database import ProductDatabase


def test_config_paths_exist_as_strings():
    assert isinstance(config.IMAGES_DIR, str)
    assert isinstance(config.MODELS_DIR, str)
    assert config.EMBEDDING_DIM == 2048


def test_database_crud(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = ProductDatabase(db_path)
    assert db.count() == 0

    rows = [
        (0, "p1", "data/images/bags/bag1.jpg", "bags", 499.0),
        (1, "p2", "data/images/bags/bag2.jpg", "bags", 599.0),
        (2, "p3", "data/images/shoes/shoe1.jpg", "shoes", 1299.0),
    ]
    db.bulk_insert(rows)
    assert db.count() == 3

    fetched = db.get_by_vector_indices([2, 0])
    fetched_ids = [r[1] for r in fetched]
    assert fetched_ids == ["p3", "p1"]  # order preserved as requested

    categories = db.get_categories()
    assert set(categories) == {"bags", "shoes"}


def test_faiss_index_roundtrip(tmp_path):
    import faiss
    dim = 16
    vectors = np.random.rand(10, dim).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = str(tmp_path / "test.index")
    faiss.write_index(index, index_path)
    loaded = faiss.read_index(index_path)

    scores, indices = loaded.search(vectors[0:1], 3)
    assert indices[0][0] == 0  # nearest neighbor of itself is itself
    assert scores[0][0] > 0.99
