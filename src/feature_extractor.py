"""
Deep visual feature extractor.

Uses a pretrained ResNet50 (ImageNet weights) with the final classification
layer removed, exposing the 2048-dim global-average-pooled embedding. This is
the "Step 1: Image Feature Extraction" stage of the pipeline.
"""
import logging

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ResNet50FeatureExtractor:
    """Extracts L2-normalized embeddings from product images using ResNet50."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading ResNet50 (pretrained) on device=%s", self.device)

        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            base_model = models.resnet50(weights=weights)
        except Exception as exc:  # pragma: no cover - fallback for offline/old torchvision
            logger.warning("Falling back to pretrained=True due to: %s", exc)
            base_model = models.resnet50(pretrained=True)

        # Remove the final FC classification layer -> keep up to global avg pool
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_image(self, image_path: str):
        try:
            img = Image.open(image_path).convert("RGB")
            return self.transform(img)
        except (FileNotFoundError, OSError) as exc:
            logger.error("Could not read image '%s': %s", image_path, exc)
            return None

    @torch.no_grad()
    def extract_batch(self, image_paths):
        """
        Extract embeddings for a batch of image paths.
        Returns (embeddings: np.ndarray [N, 2048], valid_paths: list[str])
        Skips unreadable images gracefully.
        """
        tensors, valid_paths = [], []
        for path in image_paths:
            tensor = self._load_image(path)
            if tensor is not None:
                tensors.append(tensor)
                valid_paths.append(path)

        if not tensors:
            return None, []

        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch)               # [N, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [N, 2048]
        features = torch.nn.functional.normalize(features, p=2, dim=1)  # L2 normalize
        return features.cpu().numpy(), valid_paths

    @torch.no_grad()
    def extract_single(self, image_path: str):
        """Extract a single embedding, or None if the image can't be read."""
        embeddings, valid_paths = self.extract_batch([image_path])
        if embeddings is None or len(valid_paths) == 0:
            return None
        return embeddings[0]
