"""Privacy-minimized, verified-feedback learning for gymsens.

A prediction is NEVER a training label. Only independently verified feedback
attached to a specific observation_id can train a model. The model is a
conservative *second opinion*: it may block an unsafe 'free', but never
invent a 'free' state when rules say occupied/unknown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid
import time

FEATURE_NAMES = (
    "person_count", "person_dwell_s", "time_since_person_s",
    "item_count", "stable_item_count", "items_alone_s", "occluded",
    "ambiguous_items", "time_since_claim_s", "clear_duration_s",
    "rule_free", "rule_occupied", "rule_unknown",
)
VALID_STATES = {"occupied", "free", "unknown"}


def feature_vector(runtime, now: float) -> dict[str, float]:
    def elapsed(start, default=0):
        return min(300., max(0., now - start)) if start is not None else float(default)
    person_recent = runtime.person_last_seen is not None and elapsed(runtime.person_last_seen) <= 1.5
    return {
        "person_count": float(min(10, runtime.visible_people)),
        "person_dwell_s": elapsed(runtime.person_since) if person_recent else 0.,
        "time_since_person_s": elapsed(runtime.person_last_seen, 300),
        "item_count": float(min(10, len(runtime.visible_items))),
        "stable_item_count": float(min(10, len(runtime.item_registry.stable(now)))),
        "items_alone_s": elapsed(runtime.items_only_since),
        "occluded": float(runtime.occluded),
        "ambiguous_items": float(runtime.ambiguous_items),
        "time_since_claim_s": elapsed(runtime.last_claim_at, 300),
        "clear_duration_s": elapsed(runtime.clear_since),
        "rule_free": float(runtime.state == "free"),
        "rule_occupied": float(runtime.state == "occupied"),
        "rule_unknown": float(runtime.state == "unknown"),
    }


def init_db(db_path: str | Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
            station_id TEXT NOT NULL, rule_state TEXT NOT NULL,
            predicted_state TEXT NOT NULL, features TEXT NOT NULL,
            model_version TEXT, verified_state TEXT,
            verified_at TEXT, verifier TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS observations_labeled ON observations(verified_state, recorded_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS observations_station ON observations(station_id, recorded_at)")


def record_observation(db_path, station_id, rule_state, predicted_state,
                       features, model_version=None, recorded_at=None):
    if rule_state not in VALID_STATES or predicted_state not in VALID_STATES:
        raise ValueError("Invalid availability state")
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("Feature schema mismatch")
    init_db(db_path)
    event_id = uuid.uuid4().hex
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.execute("""INSERT INTO observations
            (id, recorded_at, station_id, rule_state, predicted_state, features, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event_id, timestamp, station_id, rule_state, predicted_state,
             json.dumps(features, sort_keys=True), model_version))
    return event_id


def verify_observation(db_path, observation_id: str, state: str, verifier: str = "staff"):
    """Attach a ground-truth label from authorized on-site verification.

    Immutable once verified: relabeling requires a separate reviewed process.
    """
    if state not in {"free", "occupied", "unknown"}:
        raise ValueError("Invalid verified state")
    if not observation_id or len(observation_id) > 128 or not verifier or len(verifier) > 64:
        raise ValueError("Invalid observation/verifier")
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        found = conn.execute("SELECT recorded_at, verified_state FROM observations WHERE id=?", (observation_id,)).fetchone()
        if found is None:
            raise LookupError("Observation not found")
        if found[1] is not None:
            raise ValueError("Observation already verified (immutable)")
        # Feedback submitted for old events is allowed for audit, but labels
        # must be independently verified at the observation's timestamp.
        conn.execute("UPDATE observations SET verified_state=?, verified_at=?, verifier=? WHERE id=?",
                     (state, datetime.now(timezone.utc).isoformat(), verifier, observation_id))
    return {"observation_id": observation_id, "verified_state": state}


def review_queue(db_path, limit=20):
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT id, recorded_at, station_id, rule_state,
               predicted_state, model_version FROM observations
               WHERE verified_state IS NULL AND (predicted_state='unknown' OR predicted_state != rule_state)
               ORDER BY recorded_at DESC LIMIT ?""", (min(max(1, limit), 100),)).fetchall()
    return [dict(x) for x in rows]


def prune_unverified(db_path, days: int = 14):
    if not 1 <= days <= 365:
        raise ValueError("days must be 1..365")
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        cur = conn.execute("""DELETE FROM observations WHERE verified_state IS NULL
            AND recorded_at < strftime('%Y-%m-%dT%H:%M:%S', 'now', ?)""", (f"-{days} days",))
        return cur.rowcount


def conservative_decision(rule_state: str, probability_occupied: float | None):
    """Learning can veto 'free', never override a rule-based hold to free."""
    if rule_state != "free" or probability_occupied is None or not (0 <= probability_occupied <= 1):
        return rule_state
    if probability_occupied >= .65:
        return "occupied"
    if probability_occupied >= .35:
        return "unknown"
    return "free"


class LearningModel:
    def __init__(self, bundle=None):
        self.bundle = bundle
        if bundle and tuple(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Incompatible learned feature schema")

    @classmethod
    def load(cls, model_path):
        path = Path(model_path)
        if not path.is_file():
            return cls()
        # Joblib uses pickle: this file MUST be trusted, locally managed.
        import joblib
        return cls(joblib.load(path))

    @property
    def version(self):
        return self.bundle.get("version") if self.bundle else None

    def predict_occupied(self, features):
        if self.bundle is None:
            return None
        if set(features) != set(FEATURE_NAMES):
            raise ValueError("Incompatible observation features")
        x = [[float(features[name]) for name in FEATURE_NAMES]]
        proba = self.bundle["model"].predict_proba(x)[0]
        labels = list(self.bundle["model"].classes_)
        return float(proba[labels.index(1)])

    def apply(self, rule_state, features):
        return conservative_decision(rule_state, self.predict_occupied(features))


def _metrics(true_labels, baseline, candidate):
    def calculate(pred):
        false_free = sum(a == "occupied" and p == "free" for a, p in zip(true_labels, pred))
        true_free = sum(a == "free" and p == "free" for a, p in zip(true_labels, pred))
        unknown = sum(p == "unknown" for p in pred)
        return {"false_free": false_free, "true_free": true_free, "unknown": unknown,
                "free_precision": round(true_free / (true_free + false_free), 4)
                if true_free + false_free else None}
    return {"baseline": calculate(baseline), "candidate": calculate(candidate)}


def train_candidate(db_path, output_dir, min_labels=60, auto_promote=False):
    """Chronological holdout. Safe auto-promotion requires multi-day labels and
    a strictly reduced false-free count without excessive extra unknown states.
    No raw frames or unlabeled predictions ever enter the training set.
    """
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        rows = conn.execute("""SELECT recorded_at, rule_state, verified_state, features
            FROM observations WHERE verified_state IN ('free', 'occupied')
            ORDER BY recorded_at ASC, id ASC""").fetchall()
    if len(rows) < min_labels:
        raise ValueError(f"Only {len(rows)} verified labels, need {min_labels}")
    pivot = int(len(rows) * .8)
    train, holdout = rows[:pivot], rows[pivot:]
    ytrain = [int(r[2] == "occupied") for r in train]
    ytest = [int(r[2] == "occupied") for r in holdout]
    if len(set(ytrain)) < 2 or len(set(ytest)) < 2 or min(ytrain.count(0), ytrain.count(1)) < 10:
        raise ValueError("Need both occupied/free labels in training and chronological holdout")
    import numpy as np
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    def matrix(records):
        return np.asarray([[float(json.loads(r[3])[k]) for k in FEATURE_NAMES] for r in records], dtype=float)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
    model.fit(matrix(train), ytrain)
    probabilities = model.predict_proba(matrix(holdout))[:, list(model.classes_).index(1)]
    baseline = [r[1] for r in holdout]
    candidate = [conservative_decision(r, float(p)) for r, p in zip(baseline, probabilities)]
    metrics = _metrics([r[2] for r in holdout], baseline, candidate)
    extra_unknown = metrics["candidate"]["unknown"] - metrics["baseline"]["unknown"]
    safe = (metrics["candidate"]["false_free"] < metrics["baseline"]["false_free"]
            and extra_unknown <= max(2, math.ceil(.2 * len(holdout))))
    train_days = {r[0][:10] for r in train}
    test_days = {r[0][:10] for r in holdout}
    independent_days = bool(train_days and test_days and max(train_days) < min(test_days))
    # Same-day holdout can be used to explore, but cannot promote unattended.
    safely_promotable = safe and independent_days
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {"version": version, "feature_names": FEATURE_NAMES, "model": model}
    candidate_file = output / ("candidate-" + version + ".joblib")
    with tempfile.NamedTemporaryFile(suffix=".joblib", dir=output, delete=False) as f:
        temp = f.name
    try:
        joblib.dump(bundle, temp)
        os.replace(temp, candidate_file)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    report = {"version": version, "verified_training_samples": len(train),
              "verified_holdout_samples": len(holdout), "time_split": True,
              "independent_train_holdout_days": independent_days,
              "metrics": metrics, "safe_to_promote": safely_promotable,
              "candidate_path": str(candidate_file), "auto_promoted": False,
              "note": "Only externally verified labels used. Not yet field-calibrated."}
    if auto_promote and safely_promotable:
        # Local trusted candidate. Atomic active-pointer replacement by copy.
        import shutil
        tmp_target = output / (".active-" + version + ".joblib")
        shutil.copyfile(candidate_file, tmp_target)
        os.replace(tmp_target, output / "active.joblib")
        report["auto_promoted"] = True
    (output / ("report-" + version + ".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def labeled_count(db_path):
    init_db(db_path)
    with sqlite3.connect(db_path, timeout=10) as conn:
        return conn.execute("SELECT count(*) FROM observations WHERE verified_state IN ('free','occupied')").fetchone()[0]


def watch_training(data_dir, interval_seconds=21600, min_labels=60, auto_promote=False,
                   once=False):
    """Periodic supervised retraining. Requires fresh verified feedback.

    Each tenant has an independent database/model. Never cross-train tenants.
    Only new labels trigger training, never repeated unverified predictions.
    """
    root = Path(data_dir)
    attempted_counts = {}
    while True:
        if root.exists():
            for gym_dir in sorted(root.iterdir()):
                db = gym_dir / "observations.db"
                if not gym_dir.is_dir() or not db.is_file():
                    continue
                count = labeled_count(db)
                if count < min_labels or attempted_counts.get(gym_dir) == count:
                    continue
                attempted_counts[gym_dir] = count
                try:
                    report = train_candidate(db, gym_dir / "models", min_labels, auto_promote)
                    print(json.dumps({"gym_id": gym_dir.name, "training": report}), flush=True)
                except (ValueError, OSError) as e:
                    print(json.dumps({"gym_id": gym_dir.name, "training_error": str(e)}), flush=True)
        if once:
            return
        time.sleep(max(60, interval_seconds))


def main():
    ap = argparse.ArgumentParser(description="Train gymsens only from verified station observations")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--db", required=True)
    t.add_argument("--models", required=True)
    t.add_argument("--min-labels", type=int, default=60)
    t.add_argument("--promote-if-safe", action="store_true")
    prune = sub.add_parser("prune")
    prune.add_argument("--db", required=True)
    prune.add_argument("--days", type=int, default=14)
    w = sub.add_parser("watch")
    w.add_argument("--data-dir", default="data")
    w.add_argument("--interval-seconds", type=int, default=21600)
    w.add_argument("--min-labels", type=int, default=60)
    w.add_argument("--promote-if-safe", action="store_true")
    w.add_argument("--once", action="store_true", help="Single check for tests/cron")
    args = ap.parse_args()
    if args.command == "watch":
        watch_training(args.data_dir, args.interval_seconds, args.min_labels,
                       args.promote_if_safe, args.once)
        return
    result = train_candidate(args.db, args.models, args.min_labels, args.promote_if_safe) if args.command == "train" else {"pruned": prune_unverified(args.db, args.days)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
