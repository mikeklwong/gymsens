"""Camera-independent gym station inference. No image/video storage.

All points are normalized to [0,1] so calibration survives resolution changes
at the same aspect ratio. The camera must NOT move or change optical zoom.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Optional
import json

import cv2
import numpy as np

ITEM_LABELS = {
    "backpack", "handbag", "suitcase", "bottle", "cup", "cell phone",
    "phone", "gym bag", "bag", "key", "keys", "keychain", "keyring",
}
KEY_LABELS = {"key", "keys", "keychain", "keyring"}


def norm_label(s: str) -> str:
    return str(s).strip().lower().replace("_", "-").replace("-", " ")


def point_in_polygon(poly, pt) -> bool:
    return cv2.pointPolygonTest(np.asarray(poly, np.float32),
                                (float(pt[0]), float(pt[1])), False) >= 0


def polygon_area(poly) -> float:
    return abs(float(cv2.contourArea(np.asarray(poly, np.float32))))


def polygon_centroid(poly) -> tuple[float, float]:
    coords = np.asarray(poly, np.float32)
    return (float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])))


def rect_polygon_overlap_fraction(poly, box) -> float:
    """Polygon area concealed by axis-aligned person bounding box (0..1).

    Uses polygon clipping, including non-rectangular station zones.
    Calibration polygons should be simple (non-self-intersecting).
    """
    x1, y1, x2, y2 = box
    pts = [tuple(map(float, p)) for p in poly]
    if not pts or polygon_area(poly) <= 1e-9:
        return 0.0

    def clip(points, inside, intersect):
        if not points:
            return []
        out = []
        prev = points[-1]
        for curr in points:
            a, b = inside(prev), inside(curr)
            if b and not a:
                out.append(intersect(prev, curr))
            if b:
                out.append(curr)
            if a and not b:
                out.append(intersect(prev, curr))
            prev = curr
        return out

    def edge_x(a, b, x):
        t = (x - a[0]) / (b[0] - a[0]) if abs(b[0] - a[0]) > 1e-12 else 0
        return (x, a[1] + t * (b[1] - a[1]))

    def edge_y(a, b, y):
        t = (y - a[1]) / (b[1] - a[1]) if abs(b[1] - a[1]) > 1e-12 else 0
        return (a[0] + t * (b[0] - a[0]), y)

    pts = clip(pts, lambda p: p[0] >= x1, lambda a, b: edge_x(a, b, x1))
    pts = clip(pts, lambda p: p[0] <= x2, lambda a, b: edge_x(a, b, x2))
    pts = clip(pts, lambda p: p[1] >= y1, lambda a, b: edge_y(a, b, y1))
    pts = clip(pts, lambda p: p[1] <= y2, lambda a, b: edge_y(a, b, y2))
    return min(1.0, polygon_area(pts) / polygon_area(poly)) if len(pts) >= 3 else 0.0


@dataclass(frozen=True)
class StationConfig:
    id: str
    name: str
    interaction: list[list[float]]
    belongings: list[list[float]]

    def validate(self):
        if not self.id or not self.name:
            raise ValueError("Each station needs a nonempty id and name")
        for key, poly in (("interaction", self.interaction), ("belongings", self.belongings)):
            if len(poly) < 3 or any(len(p) != 2 or any(not (0 <= v <= 1) for v in p) for p in poly):
                raise ValueError(f"Station {self.id} has invalid normalized {key} polygon")
            if polygon_area(poly) < 1e-6:
                raise ValueError(f"Station {self.id}: {key} polygon has zero/tiny area")


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    stations: list[StationConfig]

    def validate(self, image_width: int, image_height: int):
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Invalid frame dimensions")
        if not self.stations:
            raise ValueError("No stations configured; run calibrate first")
        if abs(self.width / self.height - image_width / image_height) > 0.03:
            raise ValueError("Camera aspect ratio differs from calibration. Recalibrate.")
        ids = [s.id for s in self.stations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate station ids")
        for station in self.stations:
            station.validate()


def load_calibration(path: str) -> Calibration:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != 2:
        raise ValueError("Expected v2 normalized calibration. Run `calibrate` again.")
    w, h = data["reference_width"], data["reference_height"]
    stations = []
    for s in data["stations"]:
        interaction = s["interaction"]
        stations.append(StationConfig(s["id"], s["name"], interaction,
                                      s.get("belongings", interaction)))
    calibration = Calibration(w, h, stations)
    calibration.validate(w, h)
    return calibration


def save_calibration(path: str, width: int, height: int, stations: list[StationConfig]):
    # Validate before disk writes. Atomic write is handled by the caller.
    Calibration(width, height, stations).validate(width, height)
    return {
        "version": 2, "reference_width": width, "reference_height": height,
        "stations": [{"id": s.id, "name": s.name,
                      "interaction": s.interaction, "belongings": s.belongings}
                     for s in stations],
    }


@dataclass(frozen=True)
class PersonObs:
    track_id: Optional[int]
    bbox: tuple[float, float, float, float]  # xyxy normalized
    confidence: float = 1.0

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class ItemObs:
    label: str
    bbox: tuple[float, float, float, float]  # xyxy normalized
    confidence: float = 1.0

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_iou(a, b) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, xx2-xx1) * max(0, yy2-yy1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0.0


class Association:
    def __init__(self, stations: list[StationConfig]):
        self.stations = stations
        self.track_memory: dict[int, tuple[str, float]] = {}

    def assign_people(self, people: list[PersonObs], now: float):
        assigned = {s.id: set() for s in self.stations}
        occluded = {s.id: False for s in self.stations}
        for idx, p in enumerate(people):
            who = p.track_id if p.track_id is not None else -(idx + 1)
            choices = [s for s in self.stations if point_in_polygon(s.interaction, p.foot)]
            selected = None
            if choices:
                previous = self.track_memory.get(p.track_id) if p.track_id is not None else None
                if previous and now - previous[1] <= 6 and any(s.id == previous[0] for s in choices):
                    selected = next(s for s in choices if s.id == previous[0])
                elif len(choices) == 1:
                    selected = choices[0]
                else:
                    ranked = sorted(choices, key=lambda s: hypot(p.foot[0]-polygon_centroid(s.interaction)[0],
                                                                 p.foot[1]-polygon_centroid(s.interaction)[1]))
                    a, b = ranked[:2]
                    da = hypot(p.foot[0]-polygon_centroid(a.interaction)[0],
                               p.foot[1]-polygon_centroid(a.interaction)[1])
                    db = hypot(p.foot[0]-polygon_centroid(b.interaction)[0],
                               p.foot[1]-polygon_centroid(b.interaction)[1])
                    if db - da > 0.015:
                        selected = a
                    else:
                        for s in choices:
                            occluded[s.id] = True  # ambiguous boundary
            if selected:
                assigned[selected.id].add(who)
                if p.track_id is not None:
                    self.track_memory[p.track_id] = (selected.id, now)
            # A body obscuring a station without their feet in its zone
            # is not proof of use, so the station cannot be called free.
            for s in self.stations:
                if selected and selected.id == s.id:
                    continue
                if rect_polygon_overlap_fraction(s.interaction, p.bbox) >= 0.25:
                    occluded[s.id] = True
        self.track_memory = {k: v for k, v in self.track_memory.items() if now-v[1] <= 6}
        return assigned, occluded

    def assign_items(self, items: list[ItemObs]):
        # Dedupe overlapping predictions from neighboring ROI crops.
        unique = []
        for item in sorted(items, key=lambda i: -i.confidence):
            if not any(norm_label(item.label) == norm_label(prev.label) and
                       box_iou(item.bbox, prev.bbox) >= .45 for prev in unique):
                unique.append(item)
        assigned = {s.id: [] for s in self.stations}
        ambiguous = {s.id: False for s in self.stations}
        for item in unique:
            choices = [s for s in self.stations if point_in_polygon(s.belongings, item.center)]
            if len(choices) == 1:
                assigned[choices[0].id].append(item)
            elif len(choices) > 1:
                ranked = sorted(choices, key=lambda s: hypot(item.center[0]-polygon_centroid(s.belongings)[0],
                                                                item.center[1]-polygon_centroid(s.belongings)[1]))
                a, b = ranked[:2]
                da = hypot(item.center[0]-polygon_centroid(a.belongings)[0],
                           item.center[1]-polygon_centroid(a.belongings)[1])
                db = hypot(item.center[0]-polygon_centroid(b.belongings)[0],
                           item.center[1]-polygon_centroid(b.belongings)[1])
                if db - da > .015:
                    assigned[a.id].append(item)
                else:
                    for s in choices:
                        ambiguous[s.id] = True
        return assigned, ambiguous


@dataclass
class ItemTrack:
    label: str
    position: tuple[float, float]
    anchor: tuple[float, float]
    stationary_since: float
    last_seen: float
    observations: int = 1


class ItemRegistry:
    def __init__(self, dwell_seconds: float = 5, ttl_seconds: float = 6,
                 max_match_distance: float = .025, max_stationary_drift: float = .012):
        self.tracks: list[ItemTrack] = []
        self.dwell_seconds = dwell_seconds
        self.ttl_seconds = ttl_seconds
        self.max_match_distance = max_match_distance
        self.max_stationary_drift = max_stationary_drift

    def observe(self, items: list[ItemObs], now: float):
        # Greedy matching, same class, constrained movement. Not biometric tracking.
        self.tracks = [t for t in self.tracks if now-t.last_seen <= self.ttl_seconds]
        used = set()
        for item in sorted(items, key=lambda o: -o.confidence):
            label = norm_label(item.label)
            if label not in ITEM_LABELS:
                continue
            position = item.center
            matches = [(hypot(position[0]-t.position[0], position[1]-t.position[1]), i, t)
                       for i, t in enumerate(self.tracks)
                       if i not in used and t.label == label]
            matches.sort(key=lambda m: m[0])
            if matches and matches[0][0] <= self.max_match_distance:
                _, i, track = matches[0]
                if hypot(position[0]-track.anchor[0], position[1]-track.anchor[1]) > self.max_stationary_drift:
                    track.stationary_since = now
                    track.anchor = position
                track.position = position
                track.last_seen = now
                track.observations += 1
                used.add(i)
            else:
                self.tracks.append(ItemTrack(label, position, position, now, now))
                used.add(len(self.tracks)-1)

    def candidates(self, now: float) -> list[str]:
        return sorted({t.label for t in self.tracks if now-t.last_seen <= self.ttl_seconds})

    def stable(self, now: float) -> list[str]:
        return sorted({t.label for t in self.tracks
                       if t.observations >= 2 and now-t.stationary_since >= self.dwell_seconds
                       and now-t.last_seen <= self.ttl_seconds})


@dataclass
class StationRuntime:
    config: StationConfig
    state: str = "unknown"
    reason: str = "starting"
    person_since: Optional[float] = None
    person_last_seen: Optional[float] = None
    last_claim_at: Optional[float] = None
    clear_since: Optional[float] = None
    items_only_since: Optional[float] = None
    item_registry: ItemRegistry = field(default_factory=ItemRegistry)
    visible_people: int = 0
    visible_items: list[str] = field(default_factory=list)
    occluded: bool = False
    ambiguous_items: bool = False
    learned_state: Optional[str] = None
    observation_id: Optional[str] = None
    model_version: Optional[str] = None

    def update(self, now: float, person_ids: set[int], occluded: bool = False,
               ambiguous_items: bool = False, enter_seconds: float = 3,
               exit_seconds: float = 25, free_confirm_seconds: float = 4,
               item_unattended_limit: float = 300):
        self.visible_people = len(person_ids)
        self.occluded = bool(occluded)
        self.ambiguous_items = bool(ambiguous_items)
        if person_ids:
            if self.person_since is None:
                self.person_since = now
            self.person_last_seen = now
        elif self.person_last_seen is None or now-self.person_last_seen > 1.5:
            self.person_since = None

        person_confirmed = bool(self.person_since is not None and
                                self.person_last_seen is not None and
                                now-self.person_last_seen <= 1.5 and
                                now-self.person_since >= enter_seconds)
        stable_items = self.item_registry.stable(now)
        candidate_items = self.item_registry.candidates(now)
        self.visible_items = candidate_items
        if stable_items and not person_ids:
            if self.items_only_since is None:
                self.items_only_since = now
        else:
            self.items_only_since = None
        expired_item_hold = bool(stable_items and self.items_only_since is not None and
                                 now-self.items_only_since >= item_unattended_limit)
        item_claim = bool(stable_items and not expired_item_hold)
        any_claim = person_confirmed or item_claim
        pending_evidence = bool(person_ids or candidate_items or occluded or ambiguous_items)

        if expired_item_hold and not person_confirmed:
            self.state, self.reason = "unknown", "belongings_unattended_long"
            self.clear_since = None
        elif any_claim:
            self.state = "occupied"
            self.reason = "person_and_belongings" if person_confirmed and item_claim else (
                "person_present" if person_confirmed else "belongings_present")
            self.last_claim_at = now
            self.clear_since = None
        elif pending_evidence:
            if self.state == "occupied" and self.last_claim_at is not None and \
                    now-self.last_claim_at < exit_seconds:
                self.reason = "occupied_grace_pending_evidence"
            else:
                self.state = "unknown"
                self.reason = "occluded_or_ambiguous" if occluded or ambiguous_items else "detecting_activity"
            self.clear_since = None
        elif self.state == "occupied" and self.last_claim_at is not None and \
                now-self.last_claim_at < exit_seconds:
            self.reason = "occupied_exit_grace"
        else:
            if self.clear_since is None:
                self.clear_since = now
            if now-self.clear_since >= free_confirm_seconds:
                self.state, self.reason = "free", "clear_confirmed"
            else:
                self.state, self.reason = "unknown", "checking_clear"
        return self.state


def source_timestamp(position_ms: float, frame_index: int, fps: float,
                     previous: Optional[float]) -> float:
    """Recorded-video clock: prefer source position; fall back to frame/FPS.

    Some codecs supply CAP_PROP_POS_MSEC=0, invalid timestamps, or repeats.
    """
    fallback = max(0, frame_index-1) / fps if fps > .01 else max(0, frame_index-1) / 30
    candidate = position_ms / 1000 if 0 < position_ms < 1e12 else fallback
    if previous is not None and candidate <= previous:
        candidate = max(previous + (1 / fps if fps > .01 else 1 / 30), fallback)
    return candidate
