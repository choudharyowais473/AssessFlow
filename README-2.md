# AI Interview Practice Backend

A FastAPI backend that powers an adaptive, voice-enabled mock interview / quiz platform. It fetches interview questions, scores spoken or typed answers against expected explanations using semantic similarity, adapts question difficulty in real time, transcribes live audio over WebSockets, and pulls Wikipedia context for topics on demand.

## Features

- **User authentication** — Signup and login with `bcrypt` password hashing and JWT (HS256) access tokens (`/signin`, `/login`).
- **Adaptive question fetching** — Pulls questions from QuizAPI.io for a given field/category and difficulty, encrypts the answer explanations into a `session_token` (Fernet) so the client can't tamper with or peek at expected answers (`/question_fetch`, `/start_interview`).
- **Semantic answer scoring** — Compares a candidate's answer to the expected explanation using `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings and cosine similarity, then automatically selects the next question at a harder/easier difficulty tier based on a 65% pass threshold (`/score`).
- **AI-generated fallback questions** — If the question bank runs out at the right difficulty, a local OpenAI-compatible LLM endpoint generates a new question + explanation on the fly.
- **Live speech-to-text** — A `/STT` WebSocket endpoint streams microphone audio to a `faster-whisper` (`tiny.en`) model for near real-time transcription, authenticated via JWT passed as a query parameter.
- **Wikipedia context fetch** — `/fetch_title` retrieves Wikipedia article titles/summaries relevant to the encrypted topic stored in the session token, useful for grounding interview questions.
- **Score persistence** — `/score/finalinfo` writes a user's final score and question level back to the SQLite database.
- **Rate limiting** — Sensitive endpoints (`/fetch_title`, `/question_fetch`) are limited to 5 requests/minute per user (via JWT) or IP using `slowapi`.
- **CORS enabled** — Open CORS by default for local development (restrict `allow_origins` before deploying).
- **Low-resource friendly** — Threading for NumPy/PyTorch/Whisper is capped (1–2 threads) so the service can run on modest hardware.

## Tech Stack

| Component | Library |
|---|---|
| Web framework | FastAPI |
| Auth | PyJWT, bcrypt, OAuth2PasswordBearer |
| Encryption | cryptography (Fernet) |
| Database | SQLite3 |
| Semantic similarity | sentence-transformers, scikit-learn |
| Speech-to-text | faster-whisper |
| Rate limiting | slowapi |
| LLM fallback | openai (pointed at a local OpenAI-compatible server) |
| NLP utilities | nltk (stopwords) |

## Project Structure

```
.
├── newbackend-10.py          # Main FastAPI application
├── Enviornment_Variable.env  # Environment variables (not committed)
├── Backend.db                 # SQLite database (auto-created on first run)
├── Uwais.log                  # Application log file (auto-created)
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- A local OpenAI-compatible LLM server running at `http://localhost:3000/v1` (e.g. LM Studio, Ollama with an OpenAI-compatible proxy, etc.) for the fallback question generator
- A QuizAPI.io API key (free tier available at [quizapi.io](https://quizapi.io))
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

3. **Download NLTK stopwords** (one-time)
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

4. **Create your environment file**

   Copy `.env.example` to `Enviornment_Variable.env` and fill in real values:
   ```bash
   cp .env.example Enviornment_Variable.env
   ```

5. **Start your local LLM server** (for the fallback question generator) on `http://localhost:3000/v1`. If you don't need this feature, the app still runs — it will just return a default Docker-related fallback question if the LLM call fails.

6. **Run the server**
   ```bash
   uvicorn newbackend-10:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

   > Note: the database is auto-created on first run with four seed users (`player_mid`, `player_high`, `player_new`, `testuser`) — see source for seed passwords, and remove/change these before any real deployment.

## Environment Variables

See `.env.example` for the full list. The app loads variables from a file named `Enviornment_Variable.env` (not the default `.env` — this is hardcoded in `load_dotenv()`).

| Variable | Description |
|---|---|
| `Server_key` | Secret key used to sign JWTs and derive the Fernet encryption key. Use a long, random string. |
| `API_KEY` | Your QuizAPI.io bearer token, used to fetch interview questions. |

## API Endpoints

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

- CORS is wide open (`allow_origins=["*"]`) — restrict this before production use.
- Seed users in the database use hardcoded demo passwords — remove or rotate before deployment.
- `/fetch_title` has a loop (`for i in range(len(explanation))`) that doesn't use the loop variable and will repeat the same Wikipedia query multiple times — worth reviewing.
- No Docker/FastAPI endpoint containerization yet (planned).
- Error handling on the local LLM call and Wikipedia fetch is best-effort; consider adding retries/timeouts in production.

## License

Add your preferred license here.
