"""Tenant-isolated gymsens REST integration API.

Use TLS, secret-backed API keys, reverse proxy throttling, and private edge storage.
The API deliberately returns no frames, identities, trajectories, or track IDs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gymsens_learning import review_queue, verify_observation

GYM_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
ROLE_LEVEL = {"read": 0, "reviewer": 1, "admin": 2}


class FeedbackIn(BaseModel):
    observation_id: str = Field(min_length=1, max_length=128)
    actual_state: Literal["free", "occupied", "unknown"]
    verified_by: str = Field(default="staff", min_length=1, max_length=64)


def _read_keys(raw: str):
    """JSON mapping API_KEY -> {gym_id, role}; store only hash in RAM.

    For production place JSON in a secrets manager/container secret, not in git.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise ValueError("GYMSENS_API_KEYS_JSON is not valid JSON") from e
    if not isinstance(parsed, dict):
        raise ValueError("GYMSENS_API_KEYS_JSON must be an object")
    hashed = {}
    for key, obj in parsed.items():
        if not isinstance(key, str) or len(key) < 16 or not isinstance(obj, dict):
            raise ValueError("API keys must be at least 16 characters and have gym_id/role")
        gym_id, role = obj.get("gym_id"), obj.get("role")
        if not isinstance(gym_id, str) or not GYM_PATTERN.fullmatch(gym_id) or role not in ROLE_LEVEL:
            raise ValueError("Each key needs valid gym_id and role (read/reviewer/admin)")
        hashed[hashlib.sha256(key.encode()).digest()] = {"gym_id": gym_id, "role": role}
    return hashed


def _safe_gym(gym_id: str):
    if not GYM_PATTERN.fullmatch(gym_id):
        raise HTTPException(status_code=400, detail="Invalid gym ID")
    return gym_id


def _read_status(path: Path, stale_seconds: float):
    if not path.exists():
        raise HTTPException(status_code=503, detail="No live tracker snapshot")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        if updated.tzinfo is None:
            raise ValueError("Missing timezone")
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        stale = age > stale_seconds or age < -30 or not data.get("camera_online", False)
        return data, stale
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Invalid tracker snapshot") from exc


def create_app(*, data_dir: str | Path | None = None, api_keys_json: str | None = None,
               stale_seconds: float | None = None, allowed_origins: list[str] | None = None):
    data_dir = Path(data_dir or os.environ.get("GYMSENS_DATA_DIR", "data"))
    keys = _read_keys(api_keys_json if api_keys_json is not None else os.environ.get("GYMSENS_API_KEYS_JSON", "{}"))
    stale_seconds = stale_seconds or float(os.environ.get("GYMSENS_API_STALE_SECONDS", "12"))
    allowed_origins = (allowed_origins if allowed_origins is not None else
                       [o.strip() for o in os.environ.get("GYMSENS_CORS_ORIGINS", "").split(",") if o.strip()])
    app = FastAPI(title="gymsens API", version="0.3.0",
                  description="Privacy-minimal gym equipment availability API. Statuses: free, occupied, unknown.")
    @app.middleware("http")
    async def prevent_caching(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store, private"
        return response

    if allowed_origins:
        if "*" in allowed_origins:
            raise ValueError("Wildcard CORS is not allowed; explicitly list browser app origins")
        app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                           allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])

    def auth(x_api_key: str | None = Header(default=None)):
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing API key")
        # Always hash before lookup. Content comparisons not exposed to callers.
        auth_result = keys.get(hashlib.sha256(x_api_key.encode()).digest())
        if not auth_result:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return auth_result

    def allowed(gym_id, permission, cred):
        _safe_gym(gym_id)
        # Return 404 rather than revealing a different gym exists.
        if not hmac.compare_digest(cred["gym_id"], gym_id):
            raise HTTPException(status_code=404, detail="Gym not found")
        if ROLE_LEVEL[cred["role"]] < ROLE_LEVEL[permission]:
            raise HTTPException(status_code=403, detail="Insufficient API role")

    @app.get("/healthz", tags=["system"])
    def healthz():
        return {"service": "gymsens", "status": "ok"}  # API health, not camera health

    @app.get("/v1/gyms", tags=["equipment"])
    def gyms(cred=Depends(auth)):
        return {"gyms": [{"id": cred["gym_id"]}]}

    @app.get("/v1/gyms/{gym_id}/equipment", tags=["equipment"])
    def get_equipment(gym_id: str, cred=Depends(auth)):
        allowed(gym_id, "read", cred)
        data, stale = _read_status(data_dir / gym_id / "status.json", stale_seconds)
        results = []
        for sid, station in data.get("stations", {}).items():
            state = station.get("state") if not stale else "unknown"
            if state not in ("free", "occupied", "unknown"):
                state = "unknown"
            results.append({"id": sid, "name": station.get("name", sid),
                            "availability": state, "updated_at": data["updated_at"],
                            "observation_id": None if stale else station.get("observation_id"),
                            "confidence": station.get("confidence") if not stale else None,
                            "has_detected_belongings": bool(station.get("detected_belonging_classes")) if not stale else None})
        return {"gym_id": gym_id, "camera_online": bool(data.get("camera_online")) and not stale,
                "stale": stale, "updated_at": data["updated_at"], "equipment": results}

    @app.get("/v1/gyms/{gym_id}/equipment/{equipment_id}", tags=["equipment"])
    def get_one(gym_id: str, equipment_id: str, cred=Depends(auth)):
        listing = get_equipment(gym_id, cred)
        item = next((x for x in listing["equipment"] if x["id"] == equipment_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Equipment not found")
        return {"gym_id": gym_id, **item}

    @app.get("/v1/gyms/{gym_id}/review-queue", tags=["learning"])
    def get_queue(gym_id: str, limit: int = Query(default=20, ge=1, le=100), cred=Depends(auth)):
        allowed(gym_id, "reviewer", cred)
        return {"gym_id": gym_id, "observations": review_queue(data_dir / gym_id / "observations.db", limit)}

    @app.post("/v1/gyms/{gym_id}/feedback", tags=["learning"], status_code=201)
    def feedback(gym_id: str, body: FeedbackIn, cred=Depends(auth)):
        allowed(gym_id, "reviewer", cred)
        try:
            result = verify_observation(data_dir / gym_id / "observations.db",
                                        body.observation_id, body.actual_state,
                                        body.verified_by)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {"gym_id": gym_id, **result, "learning_status": "recorded_for_future_training"}

    return app


app = create_app()
