"""Driver recognition using facenet-pytorch embeddings."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

EMBEDDINGS_PATH = Path("data/drivers.pkl")
FACE_SIZE = 160
DEFAULT_THRESHOLD = 0.75


@dataclass
class Driver:
    name: str
    embedding: np.ndarray          # (512,) L2-normalized


class DriverDatabase:
    """Pickle-based store of registered drivers."""

    def __init__(self, path: Path = EMBEDDINGS_PATH) -> None:
        self.path = path
        self.drivers: list[Driver] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                self.drivers = [
                    Driver(d["name"], np.asarray(d["embedding"], dtype=np.float32))
                    for d in data
                ]
            except Exception:
                self.drivers = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(
                [{"name": d.name, "embedding": d.embedding} for d in self.drivers],
                f,
            )

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.drivers = [d for d in self.drivers if d.name != name]
        self.drivers.append(Driver(name, embedding))
        self.save()

    def remove(self, name: str) -> None:
        self.drivers = [d for d in self.drivers if d.name != name]
        self.save()

    def list_names(self) -> list[str]:
        return [d.name for d in self.drivers]

    def __len__(self) -> int:
        return len(self.drivers)


class FaceRecognizer:
    """Face embedding (facenet) + nearest-neighbor identification."""

    def __init__(
        self,
        device: str | None = None,
        db_path: Path = EMBEDDINGS_PATH,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self.db = DriverDatabase(db_path)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        margin: float = 0.10,
    ) -> torch.Tensor | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add relative margin to include chin/forehead
        mw = int(margin * (x2 - x1))
        mh = int(margin * (y2 - y1))
        x1 = max(0, x1 - mw)
        y1 = max(0, y1 - mh)
        x2 = min(w, x2 + mw)
        y2 = min(h, y2 + mh)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))

        tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        return tensor.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_embedding(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        tensor = self._preprocess(frame_bgr, bbox)
        if tensor is None:
            return None
        emb = self.model(tensor).cpu().numpy()[0].astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        embedding: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> tuple[str | None, float]:
        """Return (name, similarity). Name is None if below threshold."""
        if not self.db.drivers:
            return None, 0.0

        best_name, best_sim = None, -1.0
        for driver in self.db.drivers:
            sim = float(np.dot(embedding, driver.embedding))
            if sim > best_sim:
                best_sim, best_name = sim, driver.name

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        embeddings: list[np.ndarray],
    ) -> bool:
        """Register a driver from one or several embeddings (averaged)."""
        valid = [e for e in embeddings if e is not None]
        if not valid:
            return False
        avg = np.mean(np.stack(valid, axis=0), axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        self.db.add(name, avg.astype(np.float32))
        return True