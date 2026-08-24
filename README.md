# Sky Recommend — Python Recommendation Engine

**Status: engineering beta.** This repository now contains a small, deterministic recommendation API rather than a placeholder service. CI validates compile, Ruff, pytest, dependency audit, Docker build, and non-root image execution. Production deployment is not verified here.

## What it does

Sky Recommend ranks candidate items against a caller-supplied preference profile using cosine similarity. It is intended as a transparent baseline recommender for feeds, marketplaces, learning content, or similar ranking experiments where deterministic behavior is more valuable than pretending to provide a trained ML platform.

Implemented endpoints:

- `GET /healthz`
- `GET /readyz`
- `POST /v1/recommend`

Example request:

```json
{
  "profile": {"chess": 1.0, "ai": 0.5},
  "items": [
    {"id": "course-1", "features": {"chess": 1.0}},
    {"id": "course-2", "features": {"music": 1.0}}
  ],
  "limit": 10
}
```

The API returns deterministic scores using the declared `cosine-similarity-v1` algorithm.

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Container

```bash
docker build -t sky-recommend .
docker run --rm -p 8000:8000 sky-recommend
```

The image runs as a non-root application user.

## Verification

```bash
python -m compileall -q main.py tests
ruff check main.py tests
pytest -q
pip-audit -r requirements.txt
docker build -t sky-recommend:ci .
test "$(docker run --rm --entrypoint id sky-recommend:ci -u)" != "0"
```

## SKYCOIN4444 integration

Keep this service independently deployable. SKYCOIN4444 modules such as feeds, marketplace, SkySchool, or content discovery can call `/v1/recommend` through an authenticated internal adapter. The caller should own identity, authorization, feature construction, privacy/consent, rate limiting, and persistence; this repository owns deterministic ranking only.

## Limits

This is not a trained ML model, collaborative-filtering platform, vector database, personalization warehouse, fairness system, or production recommendation stack. It stores no user profiles and makes no claim of recommendation quality beyond the declared deterministic algorithm.

See [`SECURITY.md`](SECURITY.md) for security boundaries.
