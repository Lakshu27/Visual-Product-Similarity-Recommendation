"""
SQLite-backed metadata store for products.

Stores one row per product image: a stable product_id, the FAISS vector index
position, the image file path, category, and optional price. Kept normalized
(single table, no redundant columns) with an index on category for fast
filtering in the Streamlit app.
"""
import sqlite3
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ProductDatabase:
    """Thin wrapper around a SQLite products table."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS products (
                        vector_index INTEGER PRIMARY KEY,
                        product_id   TEXT NOT NULL,
                        image_path   TEXT NOT NULL,
                        category     TEXT,
                        price        REAL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_category ON products(category)"
                )
        except sqlite3.Error as exc:
            logger.error("Failed to initialize database schema: %s", exc)
            raise

    def clear(self):
        """Remove all rows (used when rebuilding the index from scratch)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM products")

    def bulk_insert(self, rows):
        """
        rows: iterable of tuples
            (vector_index, product_id, image_path, category, price)
        """
        try:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO products
                    (vector_index, product_id, image_path, category, price)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            logger.info("Inserted/updated %d product rows.", len(rows) if hasattr(rows, "__len__") else -1)
        except sqlite3.Error as exc:
            logger.error("Bulk insert failed: %s", exc)
            raise

    def get_by_vector_indices(self, vector_indices):
        """Fetch product rows for a list of FAISS vector indices, preserving order."""
        if not vector_indices:
            return []
        try:
            with self._connect() as conn:
                placeholder = ",".join("?" for _ in vector_indices)
                cur = conn.execute(
                    f"SELECT vector_index, product_id, image_path, category, price "
                    f"FROM products WHERE vector_index IN ({placeholder})",
                    list(vector_indices),
                )
                rows = {r[0]: r for r in cur.fetchall()}
            # preserve the rank order that FAISS returned
            return [rows[i] for i in vector_indices if i in rows]
        except sqlite3.Error as exc:
            logger.error("Query by vector indices failed: %s", exc)
            return []

    def get_categories(self):
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
                )
                return [r[0] for r in cur.fetchall()]
        except sqlite3.Error as exc:
            logger.error("Failed to fetch categories: %s", exc)
            return []

    def count(self):
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM products")
            return cur.fetchone()[0]
