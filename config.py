"""
Central configuration for the Visual Product Similarity project.
Keep all tunable paths/parameters here so scripts stay consistent.
"""
import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")
METADATA_CSV = os.path.join(BASE_DIR, "data", "metadata.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "faiss.index")
IDS_PATH = os.path.join(MODELS_DIR, "ids.json")
DB_PATH = os.path.join(MODELS_DIR, "products.db")

# ---- Model ----
EMBEDDING_DIM = 2048  # ResNet50 penultimate layer output size
IMAGE_SIZE = 224
BATCH_SIZE = 32

# ---- Search ----
DEFAULT_TOP_K = 5

# ---- Supported image extensions ----
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
