"""
Streamlit UI: upload a product image, view Top-K visually similar products.

Run with:
    streamlit run app.py
"""
import os
import tempfile

import streamlit as st

import config
from src.search_engine import SimilaritySearchEngine

st.set_page_config(page_title="Visual Product Similarity", page_icon="🛍️", layout="wide")


@st.cache_resource(show_spinner="Loading model and similarity index...")
def load_engine():
    return SimilaritySearchEngine()


def artifacts_exist():
    return all(
        os.path.exists(p)
        for p in (config.FAISS_INDEX_PATH, config.IDS_PATH, config.DB_PATH)
    )


def main():
    st.title("🛍️ Visual Product Similarity & Recommendation")
    st.caption("Upload a product image to find visually similar items — no keywords needed.")

    if not artifacts_exist():
        st.error(
            "No index found. Add images to `data/images/` and run "
            "`python -m src.build_index` before launching the app."
        )
        return

    engine = load_engine()

    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Number of results (Top-K)", min_value=3, max_value=20, value=5)
        categories = ["All"] + engine.db.get_categories()
        category_filter = st.selectbox("Filter by category", categories)
        category_filter = None if category_filter == "All" else category_filter
        st.caption(f"Indexed products: {engine.db.count()}")

    uploaded_file = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        col_query, col_results = st.columns([1, 3])

        with col_query:
            st.subheader("Query Image")
            st.image(uploaded_file, width=220)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Searching for visually similar products..."):
                results = engine.search(tmp_path, top_k=top_k, category_filter=category_filter)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return
        finally:
            os.unlink(tmp_path)

        with col_results:
            st.subheader(f"Top {len(results)} Similar Products")
            if not results:
                st.info("No similar products found for the selected filter.")
            else:
                # Paginate results in a responsive grid instead of loading everything at once
                cols_per_row = 4
                for row_start in range(0, len(results), cols_per_row):
                    row_items = results[row_start:row_start + cols_per_row]
                    cols = st.columns(cols_per_row)  # fixed-width columns, even on a partial last row
                    for col, item in zip(cols, row_items):
                        with col:
                            if os.path.exists(item["image_path"]):
                                st.image(item["image_path"], width=150)
                            else:
                                st.warning("Image file missing.")
                            st.markdown(f"**{item['product_id']}**")
                            st.caption(f"Category: {item['category'] or 'N/A'}")
                            st.caption(f"Similarity: {item['similarity_score']:.3f}")
                            if item["price"] is not None:
                                st.caption(f"Price: {item['price']}")
    else:
        st.info("Upload an image on the left to get started.")


if __name__ == "__main__":
    main()