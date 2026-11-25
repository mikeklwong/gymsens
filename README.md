# gymsens

**Privacy-focused camera-based availability for gym equipment.** A gym-goer wants to know *can I use this rack or bench now?*, not simply whether a person appears in its video region.

> **Status: research/pilot prototype.** Verified-feedback learning pipeline and tenant-isolated REST API implemented; **not field validated**. Never advertise occupied equipment as definitely free when camera evidence is insufficient. This repo does **not** integrate directly with the proprietary Crunch Fitness app; such integration requires authorization and access to a supported partner/developer surface.

## What gymsens does

- Accepts existing IP security camera **RTSP** streams, USB webcams, or consented video files through OpenCV. IPv4 and high-resolution decoding depend on camera/NVR network access, stream settings, and the installed FFmpeg codec support.
- Requires **3840×2160 (4K)** production frames by default, with `--allow-low-res-prototype` for safe tests. 4K alone does not ensure keys or other tiny objects can be resolved.
- YOLO person tracking (ByteTrack), footpoint-based position, track-to-station association, and ambiguity/occlusion handling.
- Per-equipment belongings ROI object detection: backpacks, bags, bottles, cups, and phones. A stationary object can hold a station. **Keys are not supported by default**: use `--extra-item-model` with separately trained/validated key weights.
- Three states: `free`, `occupied`, `unknown`. Applies entry dwell, exit grace, belongings stability and abandonment/uncertainty rules, reconnection/stale-stream logic.
- Stores **only equipment states and anonymous numeric evidence**. No frames, faces, video, person IDs, or camera credentials are logged by the software. Local `--show` is an optional debugging display; check camera/NVR vendor cloud and recording settings separately.
- **Feedback-driven learning:** each observed station decision can be logged to SQLite as anonymous features. Authorized on-site staff attach independent `free`/`occupied` ground truth to specific observation IDs. A learned logistic classifier is retrained on **verified** labels and compared with the deterministic baseline on a chronological holdout. Unsafe models are not automatically promoted.
- Integration-ready **FastAPI REST API**, OpenAPI at `/openapi.json` and `/docs`, gym-level API access control, camera freshness protection, reviewer feedback endpoint, CORS allowlist, and a Node backend adapter example.

## Architecture

```text
Existing RTSP / ONVIF-capable 4K camera
  ↓ frame in memory (no recordings)
OpenCV + YOLO person tracks + belongings detections
  ↓ anonymous station evidence
Equipment association + deterministic temporal policy
  ↓ optional local learned safety veto (never unlocks occupied/unknown)
Per-gym status.json + SQLite observations (privacy-minimized)
  ↓
FastAPI REST endpoint (tenant-scoped, API key) ── gym partner's backend ── mobile app
                          ↑
          Authorized staff reviews/corrects specific observations
                          ↓
       Candidate training → held-out validation → safe model promotion
```

### What "learning" actually means

A prediction has **no built-in ground truth**. Self-training on predictions would simply reinforce mistakes. gymsens logs each classification *attempt*, but trains **only after independent verification** (e.g. gym staff on site, or consented timestamped annotation). Verified observations are stored in `data/<gym-id>/observations.db`. Training and promotion use *per-gym* data, so one gym's labels aren't silently used to train another gym. Learning never collects raw footage.

The supervised model can currently change a `free` rule result to `unknown` or `occupied` when it learns a false-free risk. It **cannot** override `occupied`/`unknown` to `free`. This conservative overlay prioritizes avoiding situations where a customer walks up to an already taken machine. The current model probabilities are **not** shown as calibrated confidence to the consumer API.

### Install locally

Python 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Windows: copy .env.example .env
```

If OpenCV requires system libraries, install those for your OS. YOLO weights may be downloaded by Ultralytics on first run; review outbound network and model licensing. Don't place RTSP credentials in Git or shell history. The CLI `--source-env` reads credentials from a named environment variable.

### Connect an existing gym security camera

A compatible camera/NVR must offer an authorized **local RTSP feed**, support a decodable video codec (H.264 preferred where available; H.265 requires FFmpeg support), and deliver a high enough **main stream**. ONVIF is useful for discovery/configuration but **is not itself a video stream consumed by this program**. Proprietary closed camera systems without RTSP access need a vendor-approved integration; no bypass is provided.

- Confirm the camera/NVR RTSP main-stream resolution is 3840×2160 or greater. Some 4K NVRs export only a low-res substream; the `probe` command will reject it for production.
- Prefer a calibrated field of view that clearly resolves each equipment station; 4K extreme-wide-angle coverage may still miss bottles or keys. Verify that people aren't hidden by racks or pillars; sometimes two cameras are better than one.
- Use a dedicated local camera account, restrict network access, disable unnecessary cloud/recording/audio, obtain gym approval and appropriate notice, and confirm local privacy laws and contracts.
- Reolink, Amcrest, and Axis are examples of camera makers with RTSP-capable models. Verify **the exact model**, RTSP firmware support, stream dimensions, and selected codec; do not assume every camera from any vendor qualifies.

```bash
# Put actual RTSP connection string in GYMSENS_RTSP_URL outside source control:
export GYMSENS_RTSP_URL='rtsp://...'
python gym_occupancy.py probe --source-env GYMSENS_RTSP_URL
python gym_occupancy.py calibrate --source-env GYMSENS_RTSP_URL --gym-id demo
python gym_occupancy.py run --source-env GYMSENS_RTSP_URL --gym-id demo
```

On Windows PowerShell: `$env:GYMSENS_RTSP_URL="rtsp://..."`. **Do not** use the default webcam `0` in production.

For a USB webcam, phone capture, or consented video *prototype*:

```bash
python gym_occupancy.py calibrate --source 0 --allow-low-res-prototype
python gym_occupancy.py run --source 0 --allow-low-res-prototype --show
python gym_occupancy.py run --source test.mp4 --allow-low-res-prototype --history-csv predictions.csv
```

Calibration draws two polygons for each station: **interaction area** (feet/body association) and **belongings area** (bench, rack side, bag floor). Station IDs must be unique and stable. Recalibrate if the camera moves, zoom changes, or aspect ratio changes. Local runtime outputs default to `data/<gym-id>/status.json`; anonymous event logs default to `data/<gym-id>/observations.db`. Add these paths to scheduled data retention/backup policies but never commit them.

Use `--extra-item-model PATH.pt` for your *own trusted* custom-trained key/bag detector; `--require-key-detector` prevents launching without proper key label weights. A distant camera cannot reliably detect keys even with custom weights: don't treat missing detections as evidence the keys aren't there.

## REST API integration (Crunch or any gym app)

This is an **adapter/API contract**, not a claim of a supported Crunch integration. Crunch or another app developer would need to approve a partner integration and call gymsens from their **own backend**. Do not embed the gym's API key in a mobile app binary. There is no need to capture member identities.

API keys are provided through `GYMSENS_API_KEYS_JSON`, mapping **long secret tokens** to a `gym_id` and role `read`, `reviewer`, or `admin`. `read` can only view availability in its own gym. `reviewer` can also submit verified corrections. The API refuses missing/invalid tokens and discloses no other tenant's equipment.

```bash
# Generate a key. Place it in your secret manager or untracked .env:
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Then set GYMSENS_API_KEYS_JSON to a JSON object mapping that key to:
# {"gym_id":"demo","role":"admin"}. Fail-closed default is {}.
export GYMSENS_DATA_DIR=data
uvicorn gymsens_api:app --host 127.0.0.1 --port 8000
```

Browse `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/openapi.json` for live OpenAPI schema and client SDK generation. Production deployment requires HTTPS, reverse proxy rate limiting, secret rotation, network access restrictions, and authentication of end users in your gym backend. Configure browser CORS origins explicitly with `GYMSENS_CORS_ORIGINS` if needed. The built-in API key role system is a **pilot authentication layer**, not a complete enterprise identity stack.

### Example response

`GET /v1/gyms/demo/equipment` with `X-API-Key`:

```json
{
  "gym_id": "demo",
  "camera_online": true,
  "stale": false,
  "updated_at": "2026-11-24T20:00:00+00:00",
  "equipment": [
    {
      "id": "squat_rack_1", "name": "Squat Rack 1",
      "availability": "occupied", "updated_at": "2026-11-24T20:00:00+00:00",
      "observation_id": "example-event-id", "confidence": null,
      "has_detected_belongings": true
    }
  ]
}
```

The timestamp, observation ID, and availability are illustrative. The real API returns **`unknown`** if the camera is offline or its snapshot is older than `GYMSENS_API_STALE_SECONDS` (default 12 seconds), even if the last file claimed `free`.

API endpoints:

| Method | Path | Minimum role | Purpose |
|---|---|---|---|
| GET | `/healthz` | public | API process health, **not** live camera health |
| GET | `/v1/gyms` | read | Gyms authorized for the supplied key |
| GET | `/v1/gyms/{gym_id}/equipment` | read | Station states and freshness |
| GET | `/v1/gyms/{gym_id}/equipment/{equipment_id}` | read | One station |
| GET | `/v1/gyms/{gym_id}/review-queue` | reviewer | Ambiguous/overridden observations to verify |
| POST | `/v1/gyms/{gym_id}/feedback` | reviewer | Independently verified ground-truth correction |

`POST /v1/gyms/demo/feedback` (with an authorized reviewer key):

```json
{
  "observation_id": "REAL_ID_FROM_LIVE_STATUS",
  "actual_state": "occupied",
  "verified_by": "on_site_staff"
}
```

A real staff member needs to know the **state at the precise observation time**, not a later guess. Feedback labels are immutable; don't use this endpoint as a generic toggle for an app user. Integrate corrections into an authenticated staff review workflow. See `examples/mobile_backend_proxy.mjs` for a minimal Node 20+ backend adapter. If Crunch doesn't expose an integration interface, you can still use the API with your **own authorized** app/dashboard; direct app modification is not supported.

## Automated improvement / learning

After **independent verified labels** accumulate, run one candidate training job:

```bash
python gymsens_learning.py train \
  --db data/demo/observations.db \
  --models data/demo/models \
  --min-labels 60 --promote-if-safe
```

The supervised logistic model trains on the earliest 80% of verified events and evaluates on the latest 20%. It requires both free and occupied samples, at least ten per class in training, a lower **false-free count** than the rule baseline on the holdout, no excessive extra unknown states, and **separate train/holdout calendar days** for automatic promotion. If those checks fail, it saves a **candidate report only** and keeps the existing model, or defaults to rules if none was promoted. Repeated events may be correlated, so day-splitting is a minimum safeguard, not a guarantee of generalization. Review the report before commercial deployment.

To periodically retrain with newly verified labels (never automatically self-label its own predictions):

```bash
python gymsens_learning.py watch --data-dir data --interval-seconds 21600 --promote-if-safe
# Or run a single cron-style check:
python gymsens_learning.py watch --data-dir data --once --promote-if-safe
```

The worker reloads promoted local model weights at most once per minute. Models are per gym at `data/<gym_id>/models/active.joblib`. **Only load locally generated/trusted model files**: joblib serialization uses pickle and can execute arbitrary code from tampered files. Use filesystem permissions and review process controls.

**Important:** The system doesn't have a way to independently verify each prediction on its own. Every attempt can create a training candidate (anonymous observation), but without staff feedback or another trusted source, it **cannot learn whether it was correct**. Pseudo-labeling predictions as truth would make it less trustworthy.

## Evaluate predictions on manually labeled video

The deterministic and learned status stream is exported by `--history-csv` in file mode. Ground truth is a CSV of `station_id,start_s,end_s,state`, with times measured in the **video clock**. Run:

```bash
python evaluate.py --predictions predictions.csv --truth example_ground_truth.csv
python -m pytest -q
```

Track **free precision** (when shown free, actually free), **false-free rate among occupied**, **unknown fraction** and end-to-end latency by station and session. Test walk-bys, rest between sets, staff cleaning, spotting, water-bottle claims, nearby racks, occlusion, disconnection, and held-out camera/day lighting. The 4K threshold is a conservative deployment gate, not proof of actual small-object accuracy.

## Docker deployment

```bash
cp .env.example .env
# Edit .env; use a strong API secret, authorized gym ID, etc.
mkdir -p data
# On Linux, data should be writable by uid 10001 in the API/trainer container:
# sudo chown -R 10001:10001 data

docker compose up -d --build api
# Optional automatic training of verified labels across gym directories:
docker compose --profile learning up -d --build trainer
```

The API container intentionally contains **no YOLO or camera access**. Run the camera inference worker separately on a machine/Jetson with network reachability to the camera, writing to the same secured `data/<gym-id>` storage. In multi-machine production, replace shared local files with a transactional pub/sub or authenticated ingestion service; do **not** mount a shared SQLite DB over unreliable network filesystems. Bind to localhost behind a TLS reverse proxy and apply rate limits. The API and trainer run as a non-root UID; ensure the mounted data directory is writable.

## Security, privacy, and limitations

- Obtain written permission before installing or using gym CCTV feeds. Comply with gym contracts, applicable privacy laws, retention rules, notice requirements, and any restrictions on gym members' images. Never monitor locker rooms or bathrooms.
- No biometric identity tracking, facial recognition, or person trajectories are exposed or persisted. Person track IDs exist **in memory** only for short-term station assignment. Anonymous station-level numeric evidence and reviewer metadata **are** stored locally until deleted.
- Keep the local RTSP video stream private (VLAN, firewall, dedicated credentials, rotate passwords). Some camera brands have cloud recording enabled independently of gymsens; disable where necessary.
- A still water bottle may not imply a claimed machine. Detection and association need equipment-specific and gym-specific testing; prolonged unattended items become `unknown`. Small keys are **not** detectable out of the box.
- Multi-gym access control is per API key and per-gym storage; this is **not** a full SSO/multi-region platform. End-user authentication belongs to the partner's backend. Do not expose credentials in mobile clients.
- Training is on anonymized evidence, not on raw images. Custom visual object detection (e.g. keys) requires separately licensed, verified labeled imagery, adequate camera resolution, and explicit consent/permission.
- Ultralytics YOLO and its weights have **AGPL-3.0 / Enterprise licensing terms**. A proprietary commercial deployment can require a license or compliance with AGPL obligations; this repository doesn't grant rights to Ultralytics software. See https://www.ultralytics.com/license .
- No project license is assigned here. Choose legal terms deliberately before publishing publicly; private GitHub storage does not automatically resolve third-party dependency obligations.

## Project files

| File | Purpose |
|---|---|
| `gym_occupancy.py` | RTSP/video/webcam calibration, inference, publishing, feedback observation capture |
| `gym_core.py` | Polygon association, temporal logic, belongings tracking |
| `gymsens_learning.py` | Privacy-safe evidence storage, staff feedback, training, safety gates, automatic retraining |
| `gymsens_api.py` | Authentication, per-gym REST endpoint, stale-safe availability, reviewer feedback |
| `evaluate.py` | Ground-truth evaluation for video-time predictions |
| `examples/mobile_backend_proxy.mjs` | Example integration through an existing app's **backend** |
| `compose.yaml`, `Dockerfile.*` | API and optional background training infrastructure |
| `tests/` | Unit and API tests with synthetic evidence |

**Release readiness:** 28+ synthetic tests passed locally, not real footage/device performance tests. A pilot must verify true equipment availability, belonging detection recall, license/privacy compliance, stream codec support, and the model's calibration/generalization across unseen days before users rely on it.
