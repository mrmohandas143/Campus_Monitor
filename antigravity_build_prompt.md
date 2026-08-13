# Campus Monitoring & Attendance System — Project Brief

## PROJECT BRIEF

A campus-deployed, real-time face recognition attendance system running on **NVIDIA Jetson Nano**.
It detects people and faces in a camera feed, identifies registered individuals, and records attendance
to SQLite + exports to Excel/spreadsheet. Operated entirely via CLI.

### Hard Constraints
| Rule | Detail |
|------|--------|
| No TensorFlow | Not installed, not imported, not used |
| No PyTorch | Not installed, not imported, not used |
| No DeepFace | Not installed, not imported, not used |
| No YOLO / Ultralytics | The `ultralytics` package is banned |
| Inference backend | **OpenCV DNN module only** (cv2.dnn, cv2.FaceDetectorYN, cv2.FaceRecognizerSF) |
| Python version | 3.6-3.8 compatible (JetPack 4.6 / 5.x) |
| pip opencv | **Do NOT pip-install opencv on Jetson Nano** — JetPack provides a CUDA-built OpenCV; pip would override it |

---

## MODELS

| Purpose | Model | Format | Source |
|---------|-------|--------|--------|
| Face detection | YuNet (face_detection_yunet_2023mar.onnx) | ONNX | OpenCV Zoo |
| Face embedding | SFace (face_recognition_sface_2021dec.onnx) | ONNX | OpenCV Zoo |
| Person detection | OpenCV HOG+SVM (built-in) | Built-in | No download |

---

## PROJECT STRUCTURE

```
Campus_Monitor/
+-- antigravity_build_prompt.md
+-- requirements.txt              # Windows dev
+-- requirements_jetson.txt       # Jetson (no opencv line)
+-- core/
¦   +-- __init__.py
¦   +-- detector.py               # Stage 1
¦   +-- embedder.py               # Stage 1
¦   +-- recognizer.py             # Stage 3
¦   +-- tracker.py                # Stage 4
+-- db/
¦   +-- __init__.py
¦   +-- schema.py                 # Stage 2
¦   +-- crud.py                   # Stage 2
+-- attendance/
¦   +-- __init__.py
¦   +-- engine.py                 # Stage 3
¦   +-- exporter.py               # Stage 4
+-- cli/
¦   +-- __init__.py
¦   +-- main.py                   # Stage 5
+-- models/
¦   +-- face_detector/
¦   ¦   +-- face_detection_yunet_2023mar.onnx
¦   +-- embedder/
¦       +-- face_recognition_sface_2021dec.onnx
+-- data/
¦   +-- faces/
¦   +-- attendance.db
+-- scripts/
¦   +-- __init__.py
¦   +-- download_models.py
+-- tests/
    +-- __init__.py
    +-- test_stage1.py
```

---

## BUILD ORDER

### Stage 1 — Detection & Embedding
**Files**: `core/detector.py`, `core/embedder.py`
**Verify**: `python tests/test_stage1.py --image <path>`
- [ ] FaceDetector detects >=1 face
- [ ] PersonDetector detects >=1 person
- [ ] FaceEmbedder produces 128-d float32 embedding
- [ ] Same-face cosine similarity >= 0.363 (SFace threshold)
- [ ] Different-face cosine similarity < 0.363

### Stage 2 — Database
**Files**: `db/schema.py`, `db/crud.py`
- [ ] Schema creates tables without error
- [ ] register/lookup/log_attendance functions work

### Stage 3 — Recognition Pipeline
**Files**: `core/recognizer.py`, `attendance/engine.py`
- [ ] Correct identity returned for registered face
- [ ] "Unknown" returned for unregistered face

### Stage 4 — Export & Tracking
**Files**: `attendance/exporter.py`, `core/tracker.py`
- [ ] Valid .xlsx file generated
- [ ] Tracker assigns consistent IDs across frames

### Stage 5 — CLI
**Files**: `cli/main.py`
- [ ] register — capture + store face + identity
- [ ] monitor — live camera loop
- [ ] export — dump to Excel
- [ ] list — print registered persons
- [ ] report — print today attendance

---

## ATTENDANCE LOGIC

- Register: N face crops -> average embedding -> store with name + reg_no
- Recognize: cosine_similarity >= 0.363 -> match
- Log once per session per person (default 5 min cooldown)
- Unknown persons counted but not named

## DATABASE SCHEMA

```sql
CREATE TABLE persons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    reg_no        TEXT UNIQUE NOT NULL,
    embedding     BLOB NOT NULL,
    face_path     TEXT,
    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER REFERENCES persons(id),
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    confidence  REAL,
    session_date DATE GENERATED ALWAYS AS (DATE(timestamp)) STORED
);
```
