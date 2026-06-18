# AssessFlow Backend

FastAPI backend for AssessFlow, an AI-powered mock interview platform. Handles question sourcing, answer scoring, adaptive difficulty, speech-to-text, and user authentication.

## Features

- **Adaptive difficulty engine** — scores each answer with SBERT semantic similarity and moves the candidate up or down the Easy/Medium/Hard ladder based on a pass threshold.
- **Keyword-overlap deduplication** — avoids serving questions too similar to the one just asked, using stopword-filtered word overlap.
- **Question shortage fallback** — when the local question pool runs dry for the target difficulty, a local LLM (OpenAI-compatible endpoint) generates a fresh question + explanation on demand.
- **Speech-to-text** — real-time audio transcription over WebSocket using faster-whisper (tiny.en, CPU, float32).
- **JWT authentication** — login issues short-lived bearer tokens; protected routes require a valid token, including the STT WebSocket.
- **SQLite persistence** — tracks per-user question level and running score, with bcrypt-hashed passwords.

## Tech Stack

- FastAPI
- SBERT (`all-MiniLM-L6-v2`) via `sentence-transformers`
- faster-whisper (`tiny.en`)
- scikit-learn (cosine similarity)
- NLTK stopwords
- SQLite3
- bcrypt + PyJWT
- QuizAPI.io (question source)
- Local OpenAI-compatible LLM endpoint (fallback question generation)

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn numpy torch sqlite3 bcrypt pyjwt python-dotenv \
            requests sentence-transformers faster-whisper scikit-learn nltk openai
```

You'll also need the NLTK stopwords corpus:

```python
import nltk
nltk.download('stopwords')
```

### 2. Environment variables

Create an `Enviornment_Variable.env` file in the project root:

```
Server_key=your-jwt-secret-key
API_KEY=your-quizapi-io-key
```

### 3. Local LLM endpoint (optional, for fallback question generation)

This is **entirely optional**. If no local LLM is reachable at startup, the backend detects this once, logs a notice, and silently uses static difficulty-matched fallback questions instead — no crashes, no retries, no slowdown for users who skip this step.

If you do want LLM-generated fallback questions, see `LLM_SETUP.md` for the full guide (Llama 3.2 1B Q4_K_M GGUF via llama-cpp-python, served on port 3000).

### 4. Run the server

```bash
uvicorn main:app --reload
```

The SQLite database (`Backend.db`) is created automatically on first run, seeded with a few test accounts (see below).

## Seeded Test Accounts

| Username      | Password         | Level | Score |
|---------------|------------------|-------|-------|
| `player_mid`  | `hash1`          | 4     | 0.50  |
| `player_high` | `hash2`          | 7     | 0.85  |
| `player_new`  | `hash3`          | 0     | 0.00  |
| `testuser`    | `4532pqsdfv@kl`  | 0     | 0.00  |

## API Endpoints

### `POST /login`
Authenticates a user and returns a JWT bearer token (15-minute expiry).

**Request:**
```json
{ "username": "testuser", "password": "4532pqsdfv@kl" }
```

**Response:**
```json
{ "status": "Valid", "access_token": "...", "token_type": "bearer" }
```

### `POST /question_fetch` / `POST /start_interview`
Fetches a pool of questions from QuizAPI.io for the given field. `/start_interview` currently wraps `/question_fetch` directly.

**Request:**
```json
{ "Field": "javascript", "Difficulty": "easy" }
```

**Response:**
```json
{
  "questions": ["..."],
  "explanation": ["..."],
  "difficulty": ["EASY", "MEDIUM", "..."]
}
```

### `POST /score` *(requires auth)*
Scores the candidate's answer against the expected explanation, then selects the next question — bumping difficulty up on a pass, down on a fail, and falling back to LLM-generated questions if the local pool is exhausted at the target tier.

**Request:**
```json
{
  "answer": "...",
  "explanation": "...",
  "full_explanation": ["..."],
  "questions": ["..."],
  "difficulty": ["EASY", "MEDIUM", "..."],
  "current_difficulty": "EASY",
  "previous_question": "..."
}
```

**Response:**
```json
{
  "current_result": 78.42,
  "next_question": "...",
  "next_explanation": "...",
  "difficulty": "MEDIUM",
  "questions": ["..."],
  "explanation": ["..."],
  "difficulty_list": ["..."]
}
```

### `WS /STT` *(requires auth)*
Streams 16kHz mono PCM audio (Int16) and returns the transcribed text once the client stops sending data (5s silence timeout).

**Connect:**
```
ws://localhost:8000/STT?mic_status=true&token=<jwt>
```

**Response:**
```json
{ "text": "transcribed answer text", "error": null }
```

### `POST /finalinfo` *(requires auth)*
Persists the candidate's final score and question level to the database.

**Request:**
```json
{ "totalinfo": { "username": "testuser", "score": 0.78, "question_lvl": 5 } }
```

**Response:**
```json
{ "status": "successful" }
```

## Notes / Known Limitations

- Database write-back in `/score` is intentionally left empty — persistence is handled separately via `/finalinfo`.
- `/start_interview` does not yet issue a session ID; session/state management is handled client-side for now.
- Vision-based proctoring (YOLO, MediaPipe gaze tracking) and TTS are out of scope for this backend and are handled on the frontend.
- Assessment clustering (KMeans/XGBoost) and Wikipedia link resolution are intentionally excluded from current scope.
- This backend is under active development; error handling around the LLM fallback path has been hardened to degrade gracefully when no local LLM is present.
