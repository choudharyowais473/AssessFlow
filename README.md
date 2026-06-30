# AssessFlow (ProctorVision)

An AI-powered exam proctoring and adaptive assessment system. It combines a FastAPI backend (user auth, adaptive question delivery, answer scoring, live speech-to-text) with a standalone local assessment engine that runs computer-vision proctoring, audio capture/transcription, text-to-speech, and clustering-based performance analysis.

The project has two main components:

1. **Backend API** (`main.py`) — a web-facing FastAPI service for auth, adaptive question delivery, and answer scoring.
2. **Assessment Engine** (`assessment_system.py`) — a local, end-to-end proctored assessment runner that asks questions out loud, watches the webcam for violations, listens for and scores spoken answers, and clusters results for performance analysis.

## Features

### Backend API (FastAPI)
- **User authentication** — Signup and login with `bcrypt` password hashing and JWT (HS256) access tokens.
- **Adaptive question fetching** — Pulls questions from QuizAPI.io for a given field/category and difficulty, and encrypts the answer explanations into a `session_token` (Fernet) so the client can't tamper with or see expected answers.
- **Semantic answer scoring** — Compares a candidate's answer to the expected explanation using `sentence-transformers` embeddings and cosine similarity, then automatically selects the next question at a harder/easier difficulty tier based on a pass threshold.
- **AI-generated fallback questions** — When the question bank runs out at the right difficulty, a local OpenAI-compatible LLM endpoint generates a new question and explanation on the fly.
- **Live speech-to-text over WebSocket** — Streams microphone audio to a `faster-whisper` model for near real-time transcription, authenticated via JWT.
- **Wikipedia context fetch** — Retrieves Wikipedia titles/summaries relevant to a topic, useful for grounding interview/exam questions.
- **Score persistence** — Writes a user's final score and question level back to a SQLite database.
- **Rate limiting** — Sensitive endpoints are rate-limited per user/IP using `slowapi`.
- **Low-resource friendly** — Threading for NumPy/PyTorch/Whisper is capped so the service can run on modest hardware.

### Assessment Engine (Local, multi-process)
- **Webcam-based gaze tracking** — Uses MediaPipe FaceMesh to estimate eye/pupil position and classify gaze direction (center, left, right, up, down) in real time.
- **Violation detection** — Flags sustained off-screen gaze (timed) as a violation, and uses YOLO (Ultralytics) to detect prohibited objects (e.g. phones, additional people) in frame, logging instances by type.
- **Speech capture and transcription** — Captures live microphone audio via `PyAudio` and transcribes it in real time using Vosk, auto-stopping after a period of silence.
- **Text-to-speech question delivery** — Speaks questions aloud asynchronously (macOS `say` command) so the assessment can run hands-free.
- **Semantic answer similarity scoring** — Same `sentence-transformers` + cosine similarity approach as the backend, used to score the transcribed spoken answer against the expected explanation.
- **Question fetching** — Pulls questions and explanations from QuizAPI.io by category and difficulty.
- **Concurrent vision monitoring during Q&A** — Runs gaze/object detection in a separate process (via `multiprocessing`) while the question is asked and the answer is captured, then collects any violations logged during that window.
- **Performance clustering** — After an assessment, runs KMeans clustering (via a scikit-learn `Pipeline` with `ColumnTransformer`/`OrdinalEncoder`) over level, anomaly count, and answer similarity to group performance into tiers (e.g. top/average/poor performer), reporting a silhouette score for cluster quality. The trained pipeline is saved to disk with `joblib` for reuse.
- **End-to-end orchestration** — `run_assessment()` ties the above together: fetch questions → for each question, start vision monitoring, speak the question, capture and transcribe the answer, stop monitoring, score the answer, and log violations — producing a full per-question result set.

## Tech Stack

| Component | Library |
|---|---|
| Web framework | FastAPI |
| Auth | PyJWT, bcrypt, OAuth2PasswordBearer |
| Encryption | cryptography (Fernet) |
| Database | SQLite3, pandas (for querying into DataFrames) |
| Semantic similarity | sentence-transformers, scikit-learn |
| Speech-to-text (backend) | faster-whisper |
| Speech-to-text (assessment engine) | Vosk, PyAudio |
| Text-to-speech | macOS `say` (via `os.system`) |
| Computer vision | OpenCV, MediaPipe (FaceMesh), Ultralytics YOLO |
| Clustering / ML | scikit-learn (KMeans, ColumnTransformer, OrdinalEncoder), joblib |
| Concurrency | multiprocessing |
| Rate limiting | slowapi |
| LLM fallback | openai (pointed at a local OpenAI-compatible server) |
| NLP utilities | nltk (stopwords) |
| Timing | pytimer2 |

## Project Structure

```
.
├── newbackend-10.py           # FastAPI backend (auth, question delivery, scoring, STT websocket)
├── assessment_system.py        # Standalone local assessment engine (vision, audio, TTS, clustering)
├── Enviornment_Variable.env    # Environment variables (not committed)
├── Backend.db                  # SQLite database (auto-created on first run)
├── Uwais.log                   # Application log file (auto-created)
├── kmeans_model.joblib          # Saved clustering pipeline (created after running assessment())
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- A local OpenAI-compatible LLM server running at `http://localhost:3000/v1` (e.g. LM Studio, Ollama with an OpenAI-compatible proxy) for the backend's fallback question generator
- A QuizAPI.io API key (free tier available at [quizapi.io](https://quizapi.io))
- A working webcam and microphone (for the assessment engine)
- macOS, if using the assessment engine's text-to-speech as-is (it shells out to the `say` command) — swap this out for `pyttsx3` or another TTS library on other platforms
- A downloaded Vosk model (e.g. `vosk-model-small-en-us-0.15`) placed in the project directory
- YOLO weights (`yolo11n.pt`) — downloaded automatically by Ultralytics on first run, or place manually
- NLTK stopwords corpus (downloaded once, see Setup)

## Setup

1. **Clone the repo and create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the NLTK stopwords corpus** (one-time, used by the backend)
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. **Download a Vosk model** (used by the assessment engine), e.g.:
   ```bash
   curl -O https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip
   ```

5. **Create your environment file**

   Copy `.env.example` to `Enviornment_Variable.env` and fill in real values.

6. **Start your local LLM server** (for the backend's fallback question generator) on `http://localhost:3000/v1`. The backend still runs without it — it just returns a default fallback question if the call fails.

### Running the backend API
```bash
uvicorn newbackend-10:app --reload
```
The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`. The database is auto-created on first run with seed demo users — remove or change these before any real deployment.

### Running the assessment engine
```bash
python assessment_system.py
```
This runs `run_assessment()`, which fetches questions, then for each one: speaks the question, monitors the webcam for violations, listens for and transcribes the spoken answer, scores it, and prints a results summary at the end. Set your QuizAPI key in the `API_KEY` variable (or refactor to read from the environment, see Known Limitations).

## Environment Variables

| Variable | Description |
|---|---|
| `Server_key` | Secret key used to sign JWTs and derive the Fernet encryption key (backend). Use a long, random string. |
| `API_KEY` | Your QuizAPI.io bearer token, used by both the backend and the assessment engine to fetch questions. |

The backend loads variables from a file named `Enviornment_Variable.env` (not the default `.env`) — this is hardcoded in `load_dotenv()`.

## Backend API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/signin` | None | Create a new user account |
| POST | `/login` | None | Authenticate and receive a JWT (15 min expiry) |
| POST | `/question_fetch` | None (rate-limited) | Fetch a batch of questions for a field/difficulty |
| POST | `/start_interview` | Bearer JWT | Same as above, but tied to the authenticated user |
| POST | `/score` | Bearer JWT | Score an answer and get the next adaptive question |
| POST | `/score/finalinfo` | Bearer JWT | Persist final score/level to the database |
| POST | `/fetch_title` | None (rate-limited) | Fetch Wikipedia titles/summaries for a topic |
| WS | `/STT` | JWT via query param | Stream audio for live transcription |

## Known Limitations / TODO

- The assessment engine hardcodes the QuizAPI key as a plain string in `if __name__ == '__main__'` — should be read from the environment instead.
- Text-to-speech relies on the macOS `say` command and will not work on Linux/Windows as-is.
- CORS on the backend is wide open (`allow_origins=["*"]`) — restrict this before production use.
- Seed users in the backend database use hardcoded demo passwords — remove or rotate before deployment.
- The backend's `/fetch_title` endpoint has a loop that repeats the same Wikipedia query multiple times without varying it — worth reviewing.
- The assessment engine's vision monitoring stops the whole loop the moment any YOLO detection fires on the watched classes — it doesn't distinguish between a brief false positive and a sustained violation.
- No Docker containerization yet for either component (planned).
- Error handling on the local LLM call, Wikipedia fetch, and QuizAPI calls is best-effort; consider adding retries/timeouts in production.

## License

MIT — see `LICENSE`.
