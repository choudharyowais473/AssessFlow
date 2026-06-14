from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import requests
import asyncio
from typing import List
import os
import random
from vosk import Model, KaldiRecognizer
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

start_directory = '.'
target_folder = "vosk-model-small-en-us-0.15"  # enter your path maybe you wanted this if not i have written a different algo too use that to find the model by name in directory
model = None

try:
    model = Model(target_folder)
    print("Model loaded successfully from direct path!")
except Exception:
    print(f"Could not find '{target_folder}' directly. Searching subdirectories...")

if model is None:
    for root, dirs, files in os.walk(start_directory):
        if target_folder in dirs:
            absolute_path = os.path.join(root, target_folder)
            print(f"Found model folder at: {absolute_path}")
            model = Model(absolute_path)
            print("Model loaded successfully from discovered path!")
            break


# Basemodel definition
class question_fetch(BaseModel):
    Field: str
    Difficulty: str
    API_KEY: str


class scorepayload(BaseModel):
    answer: str
    explanation: str


@app.post("/question_fetch")
def question_fetcher(question: question_fetch):
    questions_info = question.model_dump()
    query = questions_info["Field"]
    difficulty_level = questions_info["Difficulty"]
    API_KEY = questions_info["API_KEY"]

    url = "https://quizapi.io/api/v1/questions"
    query = query.strip('"')
    query = query.lower()
    difficulty_level = difficulty_level.upper()

    params = {
        "category": query,
        "difficulty": difficulty_level,
        "limit": 20,
        "offset": 0
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url=url, params=params, headers=headers, timeout=5)
        data = response.json()
        data = data.get('data')
        questions = []
        explanations = []
        try:
            sample_size = min(5, len(data))
            numbers = random.sample(range(0, len(data)), sample_size)
            for index, random_pos in enumerate(numbers):
                ques = data[random_pos].get('text')
                explanation = data[random_pos].get('explanation')
                questions.append(ques)
                explanations.append(explanation)
                print(f"{index + 1}. {ques}\n")
                print(f"{index + 1}.{explanation}\n")
            return {"questions": questions, "explanation": explanations, "error": None}
        except TypeError as e:
            return {"questions": [], "explanation": [], "error": str(e)}
        except IndexError as e:
            return {"questions": [], "explanation": [], "error": str(e)}
    except requests.exceptions.RequestException as excp:
        return {"questions": [], "explanation": [], "error": str(excp)}


def similarity(answers: List[str], explanations: List[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    answer_embeddings = model.encode(answers)
    explanation_embeddings = model.encode(explanations)
    similarity_scores = cosine_similarity(answer_embeddings, explanation_embeddings)
    print(similarity_scores.diagonal().tolist())
    return {"similarity": similarity_scores.diagonal().tolist()}


@app.post('/start_interview')
def start_session(payload: question_fetch):
    result = question_fetcher(payload)
    return result


@app.post('/score')
def score(payload: scorepayload):
    result = similarity([payload.answer], [payload.explanation])
    return result


@app.websocket("/STT")
async def speech_to_text(websocket: WebSocket):
    fs = 16000
    rec = KaldiRecognizer(model, fs)
    sentence = []
    await websocket.accept()
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if not result is None:
                        print(result['text'])
                        sentence.append(result['text'])
                else:
                    partial = json.loads(rec.PartialResult())
                    text_partial = partial.get("partial", "")
                    if text_partial:
                        print(text_partial)
                    await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                print("5 seconds of silence reached. Breaking loop.")
                break
    except WebSocketDisconnect:
        print("Client disconnected unexpectedly.")

    final_result = json.loads(rec.FinalResult())
    if final_result and final_result.get('text'):
        print(final_result['text'])
        sentence.append(final_result['text'])
        await websocket.send_json({"text": sentence})
        await websocket.close()
