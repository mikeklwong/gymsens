import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest
from fastapi.testclient import TestClient

from gym_core import StationConfig, StationRuntime
from gymsens_api import create_app, _read_keys
from gymsens_learning import (
    FEATURE_NAMES, LearningModel, conservative_decision, feature_vector,
    record_observation, review_queue, train_candidate, verify_observation,
)


def sample_features(person=0, item=0, state="free"):
    return {k: float((person if k == 'person_count' else item if k == 'stable_item_count'
                      else int(k == 'rule_' + state))) for k in FEATURE_NAMES}


def test_learned_policy_cannot_claim_free_if_rule_uncertain():
    assert conservative_decision("occupied", 0.0) == "occupied"
    assert conservative_decision("unknown", 0.0) == "unknown"
    assert conservative_decision("free", .15) == "free"
    assert conservative_decision("free", .40) == "unknown"
    assert conservative_decision("free", .9) == "occupied"


def test_evidence_features_are_anonymous_and_schema_stable():
    poly = [[.1,.1],[.3,.1],[.3,.3],[.1,.3]]
    st = StationRuntime(StationConfig("bench1", "Bench", poly, poly))
    st.update(0, {123})
    st.update(3, {123})
    data = feature_vector(st, 3)
    assert set(data) == set(FEATURE_NAMES)
    assert 123 not in data.values()
    assert data["person_count"] == 1
    assert data["person_dwell_s"] == 3


def test_only_verified_observations_are_training_rows(tmp_path):
    db = tmp_path / "observations.db"
    f = sample_features()
    event = record_observation(db, "rack1", "free", "free", f)
    assert review_queue(db) == []
    with sqlite3.connect(db) as conn:
        assert conn.execute("select count(*) from observations where verified_state is not null").fetchone()[0] == 0
    verify_observation(db, event, "occupied", "on_site_staff")
    with sqlite3.connect(db) as conn:
        assert conn.execute("select verified_state from observations").fetchone()[0] == "occupied"
    with pytest.raises(ValueError, match="already verified"):
        verify_observation(db, event, "free")
    with pytest.raises(ValueError, match="Only 1 verified"):
        train_candidate(db, tmp_path / "models", min_labels=20)


def test_api_tenant_isolation_staleness_and_feedback(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    gymdir = tmp_path / "gym_1"
    gymdir.mkdir()
    (gymdir / "status.json").write_text(json.dumps({
        "updated_at": now, "camera_online": True,
        "stations": {"bench_1": {"name": "Bench 1", "state": "free", "confidence": None,
                                  "detected_belonging_classes": ["bottle"]}}
    }))
    staff = "staff-secret-a-long-random-key"
    read = "read-secret-a-long-random-key"
    app = create_app(data_dir=tmp_path, api_keys_json=json.dumps({
        staff: {"gym_id": "gym_1", "role": "reviewer"},
        read: {"gym_id": "gym_1", "role": "read"}
    }), stale_seconds=3)
    client = TestClient(app)
    p = "/v1/gyms/gym_1/equipment"
    assert client.get(p).status_code == 401
    assert client.get(p, headers={"X-API-Key": "wrong"}).status_code == 401
    res = client.get(p, headers={"X-API-Key": read})
    assert res.status_code == 200
    assert res.json()["equipment"][0]["availability"] == "free"
    assert res.json()["equipment"][0]["has_detected_belongings"]
    assert client.get("/v1/gyms/other/equipment", headers={"X-API-Key": read}).status_code == 404
    assert client.get("/v1/gyms/gym_1/review-queue", headers={"X-API-Key": read}).status_code == 403
    oid = record_observation(gymdir / "observations.db", "bench_1", "free", "free", sample_features())
    feedback = {"observation_id": oid, "actual_state": "occupied", "verified_by": "floor_manager"}
    assert client.post("/v1/gyms/gym_1/feedback", json=feedback, headers={"X-API-Key": read}).status_code == 403
    r = client.post("/v1/gyms/gym_1/feedback", json=feedback, headers={"X-API-Key": staff})
    assert r.status_code == 201
    assert r.json()["learning_status"] == "recorded_for_future_training"
    assert client.post("/v1/gyms/gym_1/feedback", json=feedback, headers={"X-API-Key": staff}).status_code == 409
    old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    (gymdir / "status.json").write_text(json.dumps({"updated_at": old, "camera_online": True,
        "stations": {"bench_1": {"name": "Bench 1", "state": "free"}}}))
    res = client.get(p, headers={"X-API-Key": read}).json()
    assert res["stale"] is True
    assert res["equipment"][0]["availability"] == "unknown"


def test_model_training_chronological_and_no_automatic_promotion_when_same_day(tmp_path):
    db = tmp_path / "observations.db"
    for i in range(100):
        label = "occupied" if i % 2 == 0 else "free"
        # Fully labeled independent station observations; same day: must not autopromote.
        f = sample_features(person=(i % 2), state="free")
        event = record_observation(db, "rack1", "free", "free", f,
                   recorded_at=(datetime(2026, 11, 24, tzinfo=timezone.utc) + timedelta(seconds=i)).isoformat())
        verify_observation(db, event, label, "staff")
    report = train_candidate(db, tmp_path / "models", min_labels=60, auto_promote=True)
    assert report["verified_training_samples"] == 80
    assert report["verified_holdout_samples"] == 20
    assert not report["auto_promoted"]
    assert Path(report["candidate_path"]).exists()
    assert not (tmp_path / "models" / "active.joblib").exists()
    mod = LearningModel.load(report["candidate_path"])
    assert mod.version == report["version"]
    assert 0 <= mod.predict_occupied(sample_features(person=1)) <= 1


def test_invalid_key_configs():
    with pytest.raises(ValueError):
        _read_keys('{"short":{"gym_id":"gym","role":"admin"}}')
    with pytest.raises(ValueError):
        create_app(api_keys_json='{}', allowed_origins=['*'])
