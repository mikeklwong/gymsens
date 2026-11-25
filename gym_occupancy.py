#!/usr/bin/env python3
"""Gym equipment availability tracker v2 (local-only processing).

Commands:
  python gym_occupancy.py calibrate --source-env GYM_RTSP_URL
  python gym_occupancy.py run --source-env GYM_RTSP_URL
  python gym_occupancy.py run --source workout.mp4 --allow-low-res-prototype --show

Install with `pip install -r requirements.txt`.
The standard COCO detector does not recognize keys. To enable key detection,
provide custom trained weights containing a key/keys/keychain class using
--extra-item-model. This still requires sufficiently close, sharp footage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import cv2
import numpy as np

from gymsens_learning import LearningModel, feature_vector, record_observation
from gym_core import (
    Association, Calibration, ITEM_LABELS, KEY_LABELS, ItemObs, PersonObs, StationConfig,
    StationRuntime, load_calibration, norm_label, point_in_polygon,
    save_calibration, source_timestamp,
)

# Set before opening FFmpeg RTSP. Can be overridden externally.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def atomic_write_json(filename: str, payload: dict):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def video_source_arg(raw: str):
    return int(raw) if re.fullmatch(r"\d+", raw) else raw


def open_capture(source, requested_width=3840, requested_height=2160):
    # Backends differ. CAP_FFMPEG timeout parameters may not be honored by all.
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    elif isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
        except (TypeError, cv2.error):
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Often ignored by RTSP backends.
    return cap


def validate_resolution(frame, args):
    height, width = frame.shape[:2]
    valid = width >= args.min_width and height >= args.min_height
    if not valid and not args.allow_low_res_prototype:
        raise ValueError(
            f"Received {width}x{height}, minimum is {args.min_width}x{args.min_height}. "
            "Select the CAMERA MAIN stream (not a low-resolution substream). "
            "For webcam/phone experiments only, pass --allow-low-res-prototype."
        )
    if not valid:
        print(f"WARNING: {width}x{height} below production threshold; "
              "small-object availability inference is unvalidated.", file=sys.stderr)
    return width, height, valid


def norm_point(px, width, height):
    return [round(px[0] / width, 6), round(px[1] / height, 6)]


def pixel_poly(normalized_points, width, height):
    return np.array([[int(round(p[0]*width)), int(round(p[1]*height))]
                     for p in normalized_points], dtype=np.int32)


def calibrate(args, source):
    cap = open_capture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read a frame; check camera RTSP URL/credentials/codec")
    width, height, _ = validate_resolution(frame, args)
    # Preview is resized for usability, while points remain normalized.
    preview_w = min(1100, width)
    scale = preview_w / width
    preview_h = int(height*scale)
    stations: list[StationConfig] = []
    working = []
    edit_mode = "interaction"

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            working.append([min(1., max(0., x/preview_w)),
                            min(1., max(0., y/preview_h))])

    cv2.namedWindow("Gym calibration", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Gym calibration", click)
    print("CLICK zone corners. n: name/finish MACHINE zone; b: finish BELONGINGS zone "
          "for most recent station; u: undo; c: clear; s: save; q: cancel")
    print("Place MACHINE polygon over floor where its user stands. Place BELONGINGS "
          "polygon over surfaces/ground where that station's bags and bottles sit.")
    while True:
        view = cv2.resize(frame, (preview_w, preview_h))
        for st in stations:
            for poly, color in ((st.interaction, (70, 220, 70)),
                                (st.belongings, (50, 170, 240))):
                cv2.polylines(view, [pixel_poly(poly, preview_w, preview_h)],
                              True, color, 2)
            cv2.putText(view, st.name, tuple(pixel_poly(st.interaction, preview_w, preview_h)[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        for p in working:
            x, y = int(p[0]*preview_w), int(p[1]*preview_h)
            cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
        if len(working) > 1:
            cv2.polylines(view, [pixel_poly(working, preview_w, preview_h)],
                          False, (0, 0, 255), 2)
        cv2.putText(view, f"Draw: {edit_mode} | stations: {len(stations)}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        cv2.imshow("Gym calibration", view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and working:
            working.pop()
        elif key == ord("c"):
            working.clear()
        elif key == ord("n") and len(working) >= 3 and edit_mode == "interaction":
            print("Enter short name for new station (in terminal): ", end="", flush=True)
            name = input().strip()
            if not name or any(s.name.casefold() == name.casefold() for s in stations):
                print("Name must be nonempty and unique")
                continue
            sid = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            if not sid or any(s.id == sid for s in stations):
                print("Station id collision; choose a different name")
                continue
            pts = [list(p) for p in working]
            # Belongings zone initially matches interaction; user should redraw it.
            stations.append(StationConfig(sid, name, pts, pts))
            working.clear()
            edit_mode = "belongings"
            print(f"Created {name}. Draw its belongings zone and press b, or "
                  "press b without drawing to reuse interaction polygon.")
        elif key == ord("b") and edit_mode == "belongings" and stations:
            if working and len(working) < 3:
                print("Belongings polygon needs at least 3 corners")
                continue
            old = stations[-1]
            stations[-1] = StationConfig(old.id, old.name, old.interaction,
                                         [list(p) for p in working] if working else old.interaction)
            working.clear()
            edit_mode = "interaction"
        elif key == ord("s"):
            if working or edit_mode != "interaction":
                print("Finish or clear current polygon first")
                continue
            data = save_calibration(args.zones, width, height, stations)
            atomic_write_json(args.zones, data)
            print(f"Saved {len(stations)} stations in {args.zones}")
            break
        elif key in (ord("q"), 27):
            print("Cancelled calibration; nothing saved")
            break
    cv2.destroyAllWindows()


class LatestFrameReader:
    """Live only: decode continuously and retain one latest frame in RAM.

    Avoids the accumulated delay that a sequential RTSP decode+inference loop has.
    The thread reconnects after stream failures. Does not record anything.
    """
    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.seq = 0
        self.received_at = 0.
        self.online = False
        self.thread = threading.Thread(target=self._work, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _work(self):
        backoff = 1.
        while not self.stop.is_set():
            cap = open_capture(self.source)
            if not cap.isOpened():
                self._mark_offline()
                cap.release()
                self.stop.wait(backoff)
                backoff = min(backoff*2, 15)
                continue
            backoff = 1.
            try:
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self._mark_offline()
                        break
                    now = time.monotonic()
                    with self.lock:
                        self.frame = frame
                        self.seq += 1
                        self.received_at = now
                        self.online = True
            finally:
                cap.release()
            self.stop.wait(backoff)
            backoff = min(backoff*2, 15)

    def _mark_offline(self):
        with self.lock:
            self.online = False
            self.frame = None

    def snapshot(self):
        with self.lock:
            return (self.frame, self.seq, self.received_at, self.online)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)


def get_yolo_models(args):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install model dependencies: pip install -r requirements.txt") from exc
    # Separate model instances so item predict() doesn't mutate person's tracker.
    people_model = YOLO(args.model)
    item_model = YOLO(args.item_model)
    extra_item_model = YOLO(args.extra_item_model) if args.extra_item_model else None
    return people_model, item_model, extra_item_model


def person_predictions(model, frame, args) -> list[PersonObs]:
    height, width = frame.shape[:2]
    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[0], conf=args.person_conf,
                         imgsz=args.person_imgsz, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    ids = result.boxes.id
    track_ids = ids.int().cpu().tolist() if ids is not None else [None]*len(boxes)
    return [PersonObs(track_ids[i], (float(x1/width), float(y1/height),
                                      float(x2/width), float(y2/height)), float(scores[i]))
            for i, (x1, y1, x2, y2) in enumerate(boxes)]


def item_predictions(models, frame, calibration, args) -> list[ItemObs]:
    """Higher-detail per-equipment crops from ORIGINAL 4K frames.

    Cropping boosts effective object pixels versus resizing the whole gym
    image to a model's input size. It cannot recover invisible tiny objects.
    """
    height, width = frame.shape[:2]
    items = []
    crops = []
    origins = []
    for station in calibration.stations:
        poly = pixel_poly(station.belongings, width, height)
        x1 = max(0, int(poly[:, 0].min())-12)
        y1 = max(0, int(poly[:, 1].min())-12)
        x2 = min(width, int(poly[:, 0].max())+13)
        y2 = min(height, int(poly[:, 1].max())+13)
        if x2-x1 < 20 or y2-y1 < 20:
            continue
        crops.append(frame[y1:y2, x1:x2])
        origins.append((x1, y1))
    if not crops:
        return []
    for model in models:
        if model is None:
            continue
        # Class names come from the model, so custom key classes work when
        # --extra-item-model provides properly trained weights.
        names = model.names
        eligible = [int(i) for i, name in names.items()
                    if norm_label(name) in ITEM_LABELS]
        if not eligible:
            print("WARNING: item model has no supported bag/bottle/keys classes", file=sys.stderr)
            continue
        results = model.predict(crops, classes=eligible, conf=args.item_conf,
                                imgsz=args.item_imgsz, verbose=False)
        for (x0, y0), res in zip(origins, results):
            if res.boxes is None:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.int().cpu().tolist()
            for bb, score, cls in zip(boxes, scores, classes):
                bx1, by1, bx2, by2 = bb
                # Don't trust detections that have essentially no image detail.
                if (bx2-bx1) < args.min_item_pixels or (by2-by1) < args.min_item_pixels:
                    continue
                label = norm_label(names[int(cls)])
                box = ((x0+float(bx1))/width, (y0+float(by1))/height,
                       (x0+float(bx2))/width, (y0+float(by2))/height)
                items.append(ItemObs(label, box, float(score)))
    return items


def draw(frame, calibration, runtimes, people, items):
    h, w = frame.shape[:2]
    # Only display locally on request; never write frames to disk.
    canvas = cv2.resize(frame, (min(w, 1400), int(h*min(w, 1400)/w)))
    ch, cw = canvas.shape[:2]
    for st in calibration.stations:
        rt = runtimes[st.id]
        color = {"occupied": (30, 50, 230), "free": (40, 210, 70),
                 "unknown": (0, 200, 255)}[rt.state]
        cv2.polylines(canvas, [pixel_poly(st.interaction, cw, ch)], True, color, 2)
        cv2.polylines(canvas, [pixel_poly(st.belongings, cw, ch)], True, (230, 160, 50), 1)
        x, y = pixel_poly(st.interaction, cw, ch)[0]
        cv2.putText(canvas, f"{st.name}: {rt.state}", (int(x), max(16, int(y))),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    for p in people:
        x1, y1, x2, y2 = p.bbox
        cv2.circle(canvas, (int((x1+x2)*cw/2), int(y2*ch)), 3, (255, 255, 255), -1)
    for item in items:
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(canvas, (int(x1*cw), int(y1*ch)),
                      (int(x2*cw), int(y2*ch)), (210, 60, 210), 2)
        cv2.putText(canvas, item.label, (int(x1*cw), max(15, int(y1*ch))),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (210, 60, 210), 1)
    cv2.imshow("Gym v2 - local debug preview", canvas)
    return (cv2.waitKey(1) & 0xff) == ord("q")


def status_payload(calibration, runtimes, width, height, validated, camera_online,
                   *, recorded_time=None, reason=None, key_detector_configured=False):
    return {
        "schema_version": 3,
        "mode": "recorded_video" if recorded_time is not None else "live",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "camera_online": bool(camera_online),
        "camera_resolution": [width, height],
        "production_resolution_validated": bool(validated),
        "custom_key_detector_configured": bool(key_detector_configured),
        "recording_time_seconds": recorded_time,
        "warning": ("Camera not yet validated" if width == 0 or height == 0 else
                    None if validated else "Prototype resolution: small items may be missed"),
        "stations": {
            st.id: {
                "name": st.name,
                "state": (runtimes[st.id].learned_state or runtimes[st.id].state) if camera_online else "unknown",
                "rule_state": runtimes[st.id].state if camera_online else "unknown",
                "observation_id": runtimes[st.id].observation_id if camera_online else None,
                "model_version": runtimes[st.id].model_version if camera_online else None,
                "confidence": None,  # withheld until independently calibrated
                "reason": ("learned_safety_veto" if runtimes[st.id].learned_state is not None
                           and runtimes[st.id].learned_state != runtimes[st.id].state else
                           runtimes[st.id].reason) if camera_online else (reason or "camera_offline"),
                "persons_near_station": runtimes[st.id].visible_people if camera_online else None,
                "detected_belonging_classes": runtimes[st.id].visible_items if camera_online else [],
            } for st in calibration.stations
        },
        "limitations": ["No standard-COCO keys detection without a custom model",
                        "Absence of detected objects is not proof that no small items exist"],
    }


def process_frame(frame, now, calibration, association, runtimes,
                  people_model, item_models, args, last_item_check, items_cached,
                  item_ambiguous_cached):
    people = person_predictions(people_model, frame, args)
    people_for_stations, occluded = association.assign_people(people, now)
    items = items_cached
    item_ambiguous = item_ambiguous_cached
    if now-last_item_check >= args.item_interval:
        items = item_predictions(item_models, frame, calibration, args)
        items_for_stations, item_ambiguous = association.assign_items(items)
        for st in calibration.stations:
            runtimes[st.id].item_registry.observe(items_for_stations[st.id], now)
        last_item_check = now
    for st in calibration.stations:
        runtimes[st.id].update(now, people_for_stations[st.id],
                               occluded=occluded[st.id],
                               ambiguous_items=item_ambiguous[st.id],
                               enter_seconds=args.enter_seconds,
                               exit_seconds=args.exit_seconds,
                               free_confirm_seconds=args.free_confirm_seconds,
                               item_unattended_limit=args.item_unattended_limit)
    return people, items, last_item_check, item_ambiguous


class LearningSession:
    """Inference + bounded logging of evidence without collecting identifiable media."""
    def __init__(self, args):
        self.db = getattr(args, "learning_db", None)
        self.model_path = getattr(args, "learned_model", None)
        self.interval = max(.5, getattr(args, "observation_interval", 2.0))
        self.model = LearningModel.load(self.model_path) if self.model_path else LearningModel()
        self.last_recorded = {}
        self.last_state = {}
        self.last_reload = -1e9
        self.model_mtime = 0

    def update(self, runtimes, now):
        if self.model_path and now-self.last_reload >= 60:
            self.last_reload = now
            from pathlib import Path
            path = Path(self.model_path)
            mtime = path.stat().st_mtime_ns if path.is_file() else 0
            if mtime != self.model_mtime:
                self.model = LearningModel.load(path)
                self.model_mtime = mtime
        for sid, rt in runtimes.items():
            features = feature_vector(rt, now)
            predicted = self.model.apply(rt.state, features)
            rt.learned_state = predicted
            rt.model_version = self.model.version
            if self.db and (now-self.last_recorded.get(sid, -1e9) >= self.interval
                            or self.last_state.get(sid) != predicted):
                rt.observation_id = record_observation(self.db, sid, rt.state, predicted,
                                                       features, self.model.version)
                self.last_recorded[sid] = now
                self.last_state[sid] = predicted


def run_file(source, args, calibration, people_model, item_models):
    cap = open_capture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        previous = None
        frame_index = 0
        calibrated = False
        history_f = open(args.history_csv, "w", newline="", encoding="utf-8") if args.history_csv else None
        history_writer = csv.DictWriter(history_f, fieldnames=["time_s", "station_id", "predicted_state", "rule_state", "reason", "model_version"]) if history_f else None
        if history_writer:
            history_writer.writeheader()
        last_item_check = -1e9
        last_published = -1e9
        items_cached = []
        item_ambiguous = {s.id: False for s in calibration.stations}
        association = Association(calibration.stations)
        runtimes = {s.id: StationRuntime(s) for s in calibration.stations}
        learner = LearningSession(args)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not calibrated:
                w, h, validated = validate_resolution(frame, args)
                calibration.validate(w, h)
                calibrated = True
                atomic_write_json(args.status, status_payload(
                    calibration, runtimes, w, h, validated, True, recorded_time=0,
                    key_detector_configured=args.key_detector_configured))
            frame_index += 1
            now = source_timestamp(float(cap.get(cv2.CAP_PROP_POS_MSEC)), frame_index,
                                   fps, previous)
            previous = now
            people, items_cached, last_item_check, item_ambiguous = process_frame(
                frame, now, calibration, association, runtimes, people_model, item_models,
                args, last_item_check, items_cached, item_ambiguous)
            learner.update(runtimes, now)
            if now-last_published >= 1:
                atomic_write_json(args.status, status_payload(
                    calibration, runtimes, w, h, validated, True, recorded_time=now,
                    key_detector_configured=args.key_detector_configured))
                if history_writer:
                    for st in calibration.stations:
                        rt = runtimes[st.id]
                        history_writer.writerow({"time_s": round(now, 3),
                            "station_id": st.id, "predicted_state": rt.learned_state or rt.state,
                            "rule_state": rt.state, "reason": rt.reason, "model_version": rt.model_version})
                last_published = now
            if args.show and draw(frame, calibration, runtimes, people, items_cached):
                break
        if not calibrated:
            raise RuntimeError("No readable video frames")
        atomic_write_json(args.status, status_payload(
            calibration, runtimes, w, h, validated, True, recorded_time=previous,
            key_detector_configured=args.key_detector_configured))
        print(f"Processed {frame_index} frames; final snapshot: {args.status}")
    finally:
        if 'history_f' in locals() and history_f:
            history_f.close()
        cap.release()
        cv2.destroyAllWindows()


def run_live(source, args, calibration, people_model, item_models):
    reader = LatestFrameReader(source).start()
    association = Association(calibration.stations)
    runtimes = {s.id: StationRuntime(s) for s in calibration.stations}
    learner = LearningSession(args)
    w, h, validated = 0, 0, False
    last_seq = -1
    last_analyzed_source_time = -1e9
    last_item_check = -1e9
    last_processed = -1e9
    last_published = -1e9
    items_cached = []
    item_ambiguous = {s.id: False for s in calibration.stations}
    print("Monitoring camera. Only state JSON is written; frames stay in memory.")
    try:
        while True:
            frame, seq, received, online = reader.snapshot()
            now = time.monotonic()
            healthy = online and frame is not None and now-received < args.stream_stale_seconds
            if healthy and seq != last_seq and now-last_processed >= 1/max(.1, args.process_fps):
                current_h, current_w = frame.shape[:2]
                if current_w != w or current_h != h:
                    w, h, validated = validate_resolution(frame, args)
                    calibration.validate(w, h)
                    # A stream profile change invalidates learned association state.
                    association = Association(calibration.stations)
                    runtimes = {s.id: StationRuntime(s) for s in calibration.stations}
                    items_cached = []
                    item_ambiguous = {s.id: False for s in calibration.stations}
                    last_item_check = -1e9
                people, items_cached, last_item_check, item_ambiguous = process_frame(
                    frame, now, calibration, association, runtimes, people_model,
                    item_models, args, last_item_check, items_cached, item_ambiguous)
                learner.update(runtimes, now)
                last_seq = seq
                last_processed = time.monotonic()
                last_analyzed_source_time = received
                if args.show and draw(frame, calibration, runtimes, people, items_cached):
                    break
            else:
                time.sleep(.03)
            # Camera connectivity alone is not enough: inference must have
            # completed recently on a fresh source frame. A slow GPU must
            # never publish an old prediction as current availability.
            now = time.monotonic()
            live_frame, _, live_received, live_online = reader.snapshot()
            fresh_camera = live_online and live_frame is not None and \
                now-live_received < args.stream_stale_seconds
            fresh_analysis = now-last_analyzed_source_time < args.stream_stale_seconds
            if now-last_published >= 1:
                atomic_write_json(args.status, status_payload(
                    calibration, runtimes, w, h, validated,
                    fresh_camera and fresh_analysis,
                    reason="stream_or_inference_stale",
                    key_detector_configured=args.key_detector_configured))
                last_published = now
    finally:
        reader.close()
        atomic_write_json(args.status, status_payload(
            calibration, runtimes, w, h, validated, False, reason="tracker_stopped",
            key_detector_configured=args.key_detector_configured))
        cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local AI gym equipment availability tracker v2")
    ap.add_argument("mode", choices=["probe", "calibrate", "run"])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--source", default=None,
                       help="Camera index, video file, RTSP/HTTP URL (prefer --source-env for secrets)")
    group.add_argument("--source-env", help="Environment variable holding full camera URL")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--status", default=None, help="Output JSON; defaults to data/<gym-id>/status.json")
    ap.add_argument("--gym-id", default="demo", help="Tenant ID: letters, digits, underscore, hyphen")
    ap.add_argument("--data-dir", default=os.environ.get("GYMSENS_DATA_DIR", "data"))
    ap.add_argument("--learning-db", default=None, help="SQLite observations database")
    ap.add_argument("--learned-model", default=None, help="Promoted local model; never load untrusted files")
    ap.add_argument("--observation-interval", type=float, default=2.0)
    ap.add_argument("--history-csv", help="Optional recording-only station status time series for evaluation")
    ap.add_argument("--model", default="yolo11s.pt", help="COCO person model")
    ap.add_argument("--item-model", default="yolo11s.pt", help="COCO belongings model")
    ap.add_argument("--extra-item-model", help="Optional CUSTOM weights for keys/other bags")
    ap.add_argument("--require-key-detector", action="store_true",
                    help="Fail startup unless custom weights provide a key/keychain class")
    ap.add_argument("--person-imgsz", type=int, default=960)
    ap.add_argument("--item-imgsz", type=int, default=960)
    ap.add_argument("--person-conf", type=float, default=.35)
    ap.add_argument("--item-conf", type=float, default=.35)
    ap.add_argument("--min-item-pixels", type=int, default=12)
    ap.add_argument("--item-interval", type=float, default=1.5)
    ap.add_argument("--process-fps", type=float, default=4.)
    ap.add_argument("--enter-seconds", type=float, default=3.)
    ap.add_argument("--exit-seconds", type=float, default=25.)
    ap.add_argument("--free-confirm-seconds", type=float, default=4.)
    ap.add_argument("--item-unattended-limit", type=float, default=300.)
    ap.add_argument("--stream-stale-seconds", type=float, default=5.)
    ap.add_argument("--min-width", type=int, default=3840)
    ap.add_argument("--min-height", type=int, default=2160)
    ap.add_argument("--allow-low-res-prototype", action="store_true")
    ap.add_argument("--show", action="store_true", help="Show local debug video window")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.gym_id):
        ap.error("Invalid --gym-id")
    gym_dir = Path(args.data_dir) / args.gym_id
    args.status = args.status or str(gym_dir / "status.json")
    args.learning_db = args.learning_db or str(gym_dir / "observations.db")
    args.learned_model = args.learned_model or str(gym_dir / "models" / "active.joblib")
    if args.source_env:
        source_value = os.environ.get(args.source_env, "")
        if not source_value:
            ap.error(f"Environment variable {args.source_env!r} is empty or unset")
    else:
        source_value = args.source if args.source is not None else "0"
    source = video_source_arg(source_value)
    if args.mode == "probe":
        cap = open_capture(source)
        try:
            if not cap.isOpened():
                raise RuntimeError("Camera did not open. Check protocol, URL, authorization, and codec.")
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera opened but did not deliver readable frames.")
            w, h, valid = validate_resolution(frame, args)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (i*8)) & 255) for i in range(4))
            print(json.dumps({"resolution": [w,h], "fps_reported": fps,
                              "fourcc": codec, "production_resolution_validated": valid}, indent=2))
        finally:
            cap.release()
    elif args.mode == "calibrate":
        calibrate(args, source)
    else:
        calibration = load_calibration(args.zones)
        models = get_yolo_models(args)
        args.key_detector_configured = bool(models[2] and any(
            norm_label(n) in KEY_LABELS for n in models[2].names.values()))
        if args.require_key_detector and not args.key_detector_configured:
            raise RuntimeError("Key recognition requires custom --extra-item-model "
                               "weights trained with a key/keys/keychain class")
        is_file = isinstance(source, str) and Path(source).is_file()
        if isinstance(source, str) and not is_file and Path(source).suffix.lower() in                 {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            raise RuntimeError("Recording file not found. Check --source path.")
        if is_file:
            run_file(source, args, calibration, models[0], (models[1], models[2]))
        else:
            run_live(source, args, calibration, models[0], (models[1], models[2]))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
