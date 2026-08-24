from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    epochs: int
    interactions: int
    final_mse: float


class CollaborativeFilter:
    """Small deterministic matrix-factorization recommender for offline/batch use."""

    def __init__(self, num_users: int, num_items: int, dimensions: int = 10, seed: int = 4444):
        if num_users <= 0 or num_items <= 0 or dimensions <= 0:
            raise ValueError("num_users, num_items, and dimensions must be positive")
        self.num_users = num_users
        self.num_items = num_items
        self.dimensions = dimensions
        rng = np.random.default_rng(seed)
        self.user_embeddings = rng.normal(0, 0.1, (num_users, dimensions))
        self.item_embeddings = rng.normal(0, 0.1, (num_items, dimensions))
        self._seen: dict[int, set[int]] = {user_id: set() for user_id in range(num_users)}

    def _validate_interaction(self, user_id: int, item_id: int, rating: float) -> None:
        if not 0 <= user_id < self.num_users:
            raise ValueError(f"user_id out of range: {user_id}")
        if not 0 <= item_id < self.num_items:
            raise ValueError(f"item_id out of range: {item_id}")
        if not np.isfinite(rating):
            raise ValueError("rating must be finite")

    def train(self, interactions: Sequence[tuple[int, int, float]], epochs: int = 10, lr: float = 0.01) -> TrainingResult:
        if not interactions:
            raise ValueError("interactions must not be empty")
        if epochs <= 0 or lr <= 0:
            raise ValueError("epochs and lr must be positive")

        for user_id, item_id, rating in interactions:
            self._validate_interaction(user_id, item_id, rating)
            self._seen[user_id].add(item_id)

        final_mse = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for user_id, item_id, rating in interactions:
                user_vector = self.user_embeddings[user_id].copy()
                item_vector = self.item_embeddings[item_id].copy()
                prediction = float(np.dot(user_vector, item_vector))
                error = rating - prediction
                total_loss += error**2
                self.user_embeddings[user_id] += 2 * lr * error * item_vector
                self.item_embeddings[item_id] += 2 * lr * error * user_vector
            final_mse = total_loss / len(interactions)
            LOGGER.info("training_epoch", extra={"epoch": epoch + 1, "mse": final_mse})

        return TrainingResult(epochs=epochs, interactions=len(interactions), final_mse=final_mse)

    def score(self, user_id: int, item_ids: Iterable[int]) -> list[tuple[int, float]]:
        if not 0 <= user_id < self.num_users:
            raise ValueError(f"user_id out of range: {user_id}")
        scored: list[tuple[int, float]] = []
        for item_id in item_ids:
            if not 0 <= item_id < self.num_items:
                raise ValueError(f"item_id out of range: {item_id}")
            scored.append((item_id, float(np.dot(self.item_embeddings[item_id], self.user_embeddings[user_id]))))
        return scored

    def recommend(self, user_id: int, top_k: int = 5, exclude_seen: bool = True) -> list[int]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = range(self.num_items)
        if exclude_seen:
            seen = self._seen.get(user_id, set())
            candidates = [item_id for item_id in candidates if item_id not in seen]
        ranked = sorted(self.score(user_id, candidates), key=lambda row: (-row[1], row[0]))
        return [item_id for item_id, _ in ranked[:top_k]]
