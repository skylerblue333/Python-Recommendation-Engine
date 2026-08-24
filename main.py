from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Sky Recommend", version="0.1.0")


class Item(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    features: dict[str, float] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    profile: dict[str, float] = Field(default_factory=dict)
    items: list[Item] = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)


@dataclass(frozen=True)
class ScoredItem:
    id: str
    score: float


def cosine_score(profile: dict[str, float], features: dict[str, float]) -> float:
    keys = set(profile) | set(features)
    if not keys:
        return 0.0
    dot = sum(profile.get(k, 0.0) * features.get(k, 0.0) for k in keys)
    a = math.sqrt(sum(profile.get(k, 0.0) ** 2 for k in keys))
    b = math.sqrt(sum(features.get(k, 0.0) ** 2 for k in keys))
    if a == 0 or b == 0:
        return 0.0
    return dot / (a * b)


def rank(request: RecommendRequest) -> list[ScoredItem]:
    scored = [ScoredItem(item.id, cosine_score(request.profile, item.features)) for item in request.items]
    scored.sort(key=lambda item: (-item.score, item.id))
    return scored[: request.limit]


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/v1/recommend")
def recommend(
    request: RecommendRequest,
    x_request_id: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    if any(not math.isfinite(value) for value in request.profile.values()):
        raise HTTPException(status_code=422, detail="profile values must be finite")
    for item in request.items:
        if any(not math.isfinite(value) for value in item.features.values()):
            raise HTTPException(status_code=422, detail=f"features for {item.id} must be finite")

    results = rank(request)
    return {
        "request_id": x_request_id,
        "algorithm": "cosine-similarity-v1",
        "results": [{"id": item.id, "score": round(item.score, 8)} for item in results],
    }
