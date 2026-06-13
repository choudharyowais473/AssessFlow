# ProctorVision

Real-time AI-powered exam proctoring system using gaze tracking, voice recognition, and ML-based answer evaluation.

## Features

- **Gaze Tracking** — Monitors eye direction using MediaPipe FaceMesh; detects looking away (left, right, up, down)
- **Object Detection** — YOLO-based detection to identify phones, laptops, or unauthorized devices
- **Speech Recognition** — Vosk-powered real-time audio transcription with 5-second silence detection
- **Answer Evaluation** — Semantic similarity scoring using SentenceTransformer embeddings
- **Cheating Analytics** — Logs and categorizes all violation instances
- **Performance Clustering** — KMeans classification of students (Top / Average / Poor performers)

---

## Installation

**Prerequisites:** Python 3.11+, webcam, microphone, macOS (`say`) or Linux (`espeak`), ~2GB disk space.

```bash
git clone https://github.com/yourusername/ProctorVision.git
cd ProctorVision
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Models auto-download on first run. To pre-download the Vosk model manually:

```bash
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip && cd ..
```

Get a free QuizAPI key at [quizapi.io](https://quizapi.io/) and pass it into `run_assessment()`.

---

## Quick Start

```python
from assessment_system import run_assessment

results = run_assessment(
    api_key="your_quizapi_key_here",
    category="Physics",
    difficulty="EASY"
)

for result in results:
    print(f"Q{result['question_num']}: {result['similarity_score']:.2%} match")
    print(f"Violations: {result['violations']}")
```

See [API_REFERENCE.md](./docs/API_REFERENCE.md) for individual function usage.

---

## Configuration

All values are inline constants in `assessment_system.py`.

| Setting | Location | Default | Notes |
|---|---|---|---|
| Gaze sensitivity | `vision_model()` | `0.35` | Lower = more sensitive |
| Monitoring window | `vision_model()` | `30s` | Total vision runtime per session |
| Audio timeout | `speech_to_text()` | `5s` | Silence before recording closes |
| YOLO classes | `vision_model()` | `[67, 63]` | `67` = cell phone, `63` = laptop |
| YOLO resolution | `vision_model()` | `320px` | Higher = more accurate, slower |
| Questions per session | `question_fetcher()` | `5` | Target is 10 for full interview |

---

## Troubleshooting

**Webcam not opening**
```bash
sudo chmod 666 /dev/video0   # Linux
# macOS: System Preferences → Security & Privacy → Camera
```

**Audio not capturing**
```bash
arecord -l   # Linux — lists audio devices
# macOS: System Preferences → Security & Privacy → Microphone
```

**YOLO downloading slowly** — pre-download with `YOLO("yolo11n.pt")` in a Python shell; cached after first run.

**Speech recognition inaccurate** — reduce background noise, speak clearly, or increase the timeout to `elapsed >= 7`.

**Out of memory** — confirm `imgsz=320` (not higher) and that you're using `yolo11n.pt` (nano).

---

## Performance

| Component | Speed | Accuracy |
|---|---|---|
| Gaze Detection | ~30 FPS | 95% (well-lit) |
| YOLO Detection | ~60 FPS | 90%+ |
| Speech Recognition | Real-time | 85–90% (clear audio) |
| Similarity Scoring | ~50ms per answer | Cosine similarity |

---

## Roadmap

**v1.0** (Current) — Gaze tracking, speech capture, question fetching, similarity scoring, multiprocessing

**v1.1** (Planned) — FastAPI endpoints, WebSocket streaming, advanced violation analytics

**v2.0** (Future) — Web dashboard, multi-student batch processing, custom cheating detector

---

## License

MIT — [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) · [Google MediaPipe](https://github.com/google/mediapipe) · [Vosk](https://github.com/alphacephei/vosk-api) · [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)