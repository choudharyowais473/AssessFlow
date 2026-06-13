# API Reference

All functions are importable from `assessment_system`.

---

## `run_assessment(api_key, category, difficulty)`

Main orchestration function. Runs the complete assessment workflow end-to-end.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `api_key` | `str` | QuizAPI key |
| `category` | `str` | Subject — e.g. `"Physics"`, `"Math"`, `"Chemistry"` |
| `difficulty` | `str` | `"EASY"`, `"MEDIUM"`, or `"HARD"` |

**Returns:** List of result dicts — see [Output Structure](#output-structure) below.

**Example:**

```python
from assessment_system import run_assessment

results = run_assessment(
    api_key="your_key",
    category="Physics",
    difficulty="EASY"
)

for r in results:
    print(f"Q{r['question_num']}: {r['similarity_score']:.2%}")
    print(f"Violations: {r['violations']}")
```

---

## `question_fetcher(query, difficulty_level, api_key)`

Fetches a random set of questions from QuizAPI.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `query` | `str` | Subject category |
| `difficulty_level` | `str` | `"EASY"`, `"MEDIUM"`, or `"HARD"` |
| `api_key` | `str` | QuizAPI authorization key |

**Returns:**

```python
{
    "questions": ["Question 1 text", "Question 2 text", ...],
    "explanations": ["Explanation 1", "Explanation 2", ...]
}
```

---

## `vision_model(violation_queue)`

Monitors gaze direction and detects physical cheat signals via YOLO. Designed to run in a separate process via `multiprocessing`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `violation_queue` | `Queue` | Multiprocessing queue — violations are pushed here |

**Returns:**

```python
{
    "violations": {
        "Looking Right": 2,
        "Looking Left": 1
    }
}
```

**Example:**

```python
from multiprocessing import Queue, Process
from assessment_system import vision_model

violation_queue = Queue()
vision_process = Process(target=vision_model, args=(violation_queue,))
vision_process.start()

# ... interview runs here ...

vision_process.terminate()
vision_process.join()

violations = []
while not violation_queue.empty():
    violations.append(violation_queue.get_nowait())
```

---

## `speech_to_text()`

Captures audio from the microphone and returns a transcription. Uses a 5-second silence timeout.

**Returns:**

```python
{
    "text": "student's spoken response"
}
```

---

## `text_to_speech(text_to_say)`

Speaks text aloud using the system audio (`say` on macOS, `espeak` on Linux).

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `text_to_say` | `str` | The text to be spoken |

**Returns:** None

---

## `similarity(answers, explanations)`

Computes semantic similarity between student answers and reference explanations using SBERT embeddings and cosine similarity.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `answers` | `list[str]` | Student answer strings |
| `explanations` | `list[str]` | Reference/correct explanation strings |

**Returns:**

```python
{
    "similarity": [0.87, 0.92, ...]  # One score per answer-explanation pair, 0–1 scale
}
```

---

## `title_fetch(query_phrase)`

Fetches Wikipedia article titles relevant to a search query. Used to surface learning resources for weak answers.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `query_phrase` | `str` | Search term derived from the question |

**Returns:**

```python
{
    "titles": ["Title 1", "Title 2", ...]
}
```

---

## `assessment(local_db)`

Performs KMeans clustering on session data to classify student performance. Requires an existing database with an `assessment_table`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `local_db` | `sqlite3.Connection` | Active database connection |

**Returns:**

```python
{
    "silhouette_score": 0.654,
    "performance_groups": {
        "Top Performer": 12,
        "Average Performer": 8,
        "Poor Performer": 5
    }
}
```

---

## Output Structure

Each entry in the `run_assessment()` result list:

```python
{
    "question_num": 1,
    "question": "What is photosynthesis?",
    "student_answer": "Process where plants convert sunlight to energy",
    "similarity_score": 0.87,   # 0–1 scale
    "violations": ["Looking Right", "Looking Left"]
}
```

### Violation Types

| Value | Trigger |
|---|---|
| `"Looking Left"` | Sustained left gaze for 5+ seconds |
| `"Looking Right"` | Sustained right gaze for 5+ seconds |
| `"Looking Up"` | Sustained upward gaze for 5+ seconds |
| `"Looking Down"` | Sustained downward gaze for 5+ seconds |
| Object detection | Phone or unauthorized person detected — ends assessment immediately |