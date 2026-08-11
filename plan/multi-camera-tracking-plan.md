# Multi-Camera Object Tracking & Path Reconstruction — Project Plan

## 1. Concept Summary

A system that identifies a specific object (starting with vehicles via license plate) across multiple independent camera feeds — live or recorded — and reconstructs its movement path over time by correlating sightings with camera locations and synchronized timestamps.

**Primary use case:** stolen vehicle tracking — input a plate number, get back a timeline: which cameras saw it, when, and a mapped path of movement.

**Generalization path:** the same architecture works for any trackable object (person re-id, bags, other vehicle types) by swapping the identification layer.

---

## 2. System Architecture

### 2.1 Pipeline Stages

```
[Video Sources] → [Frame Sampling] → [Detection] → [Identification] → [Sighting Store]
                                                                              ↓
[Map/Timeline UI] ← [Path Reconstruction] ← [Cross-Camera Matching] ← [Sighting Store]
```

### 2.2 Core Components

| Component | Purpose | Candidate Tools |
|---|---|---|
| Frame ingestion | Pull frames from live/recorded sources | OpenCV, FFmpeg, RTSP client |
| Vehicle detection | Locate cars in frame | YOLOv8 / YOLOv11 |
| Plate detection | Locate plate region within car bbox | Dedicated plate-detector model or YOLO fine-tuned on plates |
| Plate OCR | Read plate text | PaddleOCR, EasyOCR, OpenALPR |
| Vehicle re-id embedding | Fallback visual identity when plate unreadable | ResNet/OSNet trained on VeRi-776 or VehicleID |
| Sighting database | Store every detection event | PostgreSQL + pgvector (for embeddings) |
| Matching engine | Link sightings to a target vehicle | Exact/fuzzy plate match + cosine similarity search |
| Camera topology | Known camera positions & travel-time constraints | Simple JSON config for PoC |
| Path reconstruction | Order sightings into a coherent timeline/route | Time-window + topology-constrained search |
| Visualization | Show path on map + timeline | Leaflet/Folium (map), simple web UI |

---

## 3. Data Model (PoC-level)

**Sighting record:**
```json
{
  "sighting_id": "uuid",
  "camera_id": "cam_03",
  "timestamp": "2026-08-10T14:22:11.500Z",
  "plate_text": "ABC1234",
  "plate_confidence": 0.91,
  "embedding": [0.021, -0.114, ...],
  "bbox": [x1, y1, x2, y2],
  "thumbnail_path": "storage/thumbs/cam03_142211.jpg"
}
```

**Camera record:**
```json
{
  "camera_id": "cam_03",
  "name": "Main St & 5th Ave",
  "lat": 40.7128,
  "lon": -74.0060,
  "adjacent_cameras": ["cam_01", "cam_04"],
  "min_travel_time_sec": {"cam_01": 45, "cam_04": 90}
}
```

---

## 4. Matching Logic

1. **Plate match (primary):** exact string match first; fall back to fuzzy match (edit distance ≤1–2) to tolerate OCR errors (0/O, 1/I, 8/B confusions).
2. **Re-id match (fallback):** when plate unreadable/low-confidence, compare embedding via cosine similarity against known sightings of the target; flag matches below a confidence threshold for human review rather than auto-accepting.
3. **Temporal-spatial constraint:** only consider a candidate sighting at Camera B if it falls within a plausible time window after a confirmed sighting at Camera A, based on the camera topology's travel-time estimates. This prunes false positives and keeps search cheap.
4. **Confidence scoring:** every link in the reconstructed path carries a confidence score (plate confidence × time-plausibility × embedding similarity if used) so the output timeline can visually distinguish "certain" vs. "probable" hops.

---

## 5. Live vs. Recorded Handling

- **Recorded footage:** batch process at higher frame rate/thoroughness; timestamps come from file metadata or embedded overlay OCR if needed.
- **Live feeds:** stream at reduced sampling rate (2–5 fps is usually enough for vehicle detection); push sightings into the same database in real time; UI updates path incrementally as new sightings arrive.
- Both paths converge on the same sighting schema and matching engine — no separate logic needed downstream of ingestion.

---

## 6. Timestamp Synchronization

- Cameras will have clock drift. For a PoC, manually record and apply a per-camera offset.
- Production version: NTP-sync all camera systems, or calibrate offsets using a known reference event (e.g., a test vehicle passing two cameras at a known time).

---

## 7. Proof-of-Concept Scope (Recommended Cut)

**In scope:**
- 2–4 recorded video clips (not live streams initially)
- Vehicle detection + plate OCR pipeline
- Manual camera topology config (JSON, no road-network routing)
- Plate-based matching with fuzzy tolerance
- Re-id embeddings computed and stored, used only as fallback
- Simple map + timeline visualization of one target vehicle's path

**Out of scope (defer):**
- Live RTSP ingestion
- Automatic road-network-based travel-time calculation
- Multi-object simultaneous tracking at scale
- Production-grade auth, scaling, storage infra

**Suggested test data:** VeRi-776 dataset — multi-camera vehicle re-id dataset with plate/attribute annotations, good for validating matching logic before touching real footage.

---

## 8. Build Order

1. **Sighting data layer first** — define schema, stand up the database, populate with sample/synthetic sightings.
2. **Matching + path reconstruction logic** — build and test against synthetic data so the "brain" of the system is proven before dealing with messy real-world CV.
3. **Detection + OCR pipeline** — turn real video into sighting records; this is the fiddliest part (lighting, angles, occlusion) so it's easiest to tune once the downstream logic already works.
4. **Re-id embedding fallback** — add once plate-only matching is solid.
5. **Visualization layer** — map + timeline UI, wire up to real pipeline output last.
6. **Live ingestion** — only after recorded-footage pipeline is validated end-to-end.

---

## 9. Open Decisions for Later

- Exact detection/OCR models to fine-tune vs. use off-the-shelf
- Database choice for embeddings at scale (pgvector vs. FAISS vs. Milvus) if moving beyond PoC
- How travel-time constraints get calculated in production (road network API vs. manual config)
- Human-in-the-loop review UI for low-confidence matches
