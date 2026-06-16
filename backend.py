import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel
import requests
import asyncio
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import random
from faster_whisper import WhisperModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
app = FastAPI()
model_sbert=SentenceTransformer("all-MiniLM-L6-v2")
vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
model = WhisperModel("tiny.en", device="cpu", compute_type="float32", cpu_threads=2)

class question_fetch(BaseModel):
    Field: str
    Difficulty: str
    API_KEY: str

class scorepayload(BaseModel):
    answer: str
    explanation: str
    full_explanation: List[str]
    questions: List[str] | None = None
    difficulty: List[str] | None = None
    current_difficulty: str
    previous_question: Optional[str] = ""

@app.post("/question_fetch")
def question_fetcher(question: question_fetch):
    questions_info = question.model_dump()
    query = questions_info["Field"].strip('"').lower()
    difficulty_level = questions_info["Difficulty"].upper()
    API_KEY = questions_info["API_KEY"]
    
    url = "https://quizapi.io/api/v1/questions"
    params_current = {"category": query, "limit": 50}
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.get(url=url, params=params_current, headers=headers, timeout=5)
        data = response.json()
        
        if isinstance(data, dict) and 'data' in data:
            data = data.get('data')
            
        if not data:
            return {"Error": "No questions returned"}
            
        questions_level = []
        explanations_level = []
        difficult_level = []
        
        try:
            sample_size = min(10, len(data))
            numbers = random.sample(range(0, len(data)), sample_size)   
            for index, random_pos in enumerate(numbers): 
                level = data[random_pos].get("difficulty", "Medium")  
                ques = data[random_pos].get('text', data[random_pos].get('question', ''))  
                explanation = data[random_pos].get('explanation', 'No explanation available')
                
                questions_level.append(ques)  
                explanations_level.append(explanation) 
                difficult_level.append(level)
                
            return {
                "questions": questions_level, 
                "explanation": explanations_level, 
                "difficulty": difficult_level
            }
        except (TypeError, IndexError) as e:
            return {"Error": str(e)}
    except requests.exceptions.RequestException as excp:
        return {"excp": str(excp)}

async def similarity(answers: List[str], explanations: List[str]):
    answer_vecs = await asyncio.to_thread(model_sbert.encode, answers)
    explanation_vecs = await asyncio.to_thread(model_sbert.encode, explanations)
    similarity_scores = cosine_similarity(answer_vecs, explanation_vecs)
    return {"similarity": similarity_scores.diagonal().tolist()}

@app.post('/start_interview')
def start_session(payload: question_fetch):
    return question_fetcher(payload)

@app.post('/score')
async def score(payload: scorepayload):
    result_list = []
    sim_data = (await similarity([payload.answer], [payload.explanation])).get("similarity")
    result_list.append(sim_data)
    
    final_score = float(sim_data[0])
    calculated_percentage = final_score * 100 
    
    question = payload.questions if payload.questions else []
    explanation = payload.full_explanation if payload.full_explanation else [] # Map input schema
    difficulty = payload.difficulty if payload.difficulty else []
    
    difficult_range = ["EASY", "MEDIUM", "HARD"]
    
    try:
        position = difficult_range.index(payload.current_difficulty)
    except ValueError:
        position = 1
    
    question_to_return = ''
    explanation_to_return = ''
    difficulty_to_return = ''
    found_question = False
    
    PASS_THRESHOLD = 65.0 
    
    if calculated_percentage >= PASS_THRESHOLD:
        next_tier = difficult_range[min(position + 1, len(difficult_range) - 1)]
        for index in range(len(difficulty) - 1, -1, -1):
            if difficulty[index] == next_tier:
                try:
                    vectorizer.fit([payload.previous_question])
                    words_prev = set(vectorizer.get_feature_names_out())
                    
                    vectorizer.fit([question[index]])
                    words_current = set(vectorizer.get_feature_names_out())
                    
                    intersection_w = words_prev.intersection(words_current)
                    overlap_ratio = len(intersection_w) / max(len(words_current), 1)
                    
                    if overlap_ratio > 0.30 and len(difficulty) > 1:
                        continue
                        
                    question_to_return = question.pop(index)
                    explanation_to_return = explanation.pop(index)
                    difficulty_to_return = difficulty.pop(index)
                    found_question = True
                    break
                except ValueError:
                    question_to_return = question.pop(index)
                    explanation_to_return = explanation.pop(index)
                    difficulty_to_return = difficulty.pop(index)
                    found_question = True
                    break
                    
    elif calculated_percentage < PASS_THRESHOLD:
        next_tier = difficult_range[max(position - 1, 0)]
        for index in range(len(difficulty) - 1, -1, -1):
            if difficulty[index] == next_tier:
                question_to_return = question.pop(index)
                explanation_to_return = explanation.pop(index)
                difficulty_to_return = difficulty.pop(index)
                found_question = True
                break
                
    if not found_question and question:
        question_to_return = question.pop(0)
        explanation_to_return = explanation.pop(0)
        difficulty_to_return = difficulty.pop(0)
                    
    return {
        "current_result": round(calculated_percentage, 2), 
        "next_question": question_to_return, 
        "next_explanation": explanation_to_return, 
        "difficulty": difficulty_to_return,
        "questions": question,
        "explanation": explanation,
        "difficulty_list": difficulty
    }

@app.websocket("/STT")
async def get_voice(websocket: WebSocket, mic_status: bool = Query(True)):
    await websocket.accept()
    if not mic_status:
        await websocket.send_json({"text": "", "error": "Microphone status is disabled."})
        await websocket.close()
        return
        
    sentence = b""
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5)
                sentence += data
            except asyncio.TimeoutError:
                break
    except WebSocketDisconnect:
        pass
        
    if len(sentence) > 0:
        try:
            audio_array = np.frombuffer(sentence, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = await asyncio.to_thread(model.transcribe, audio_array, beam_size=3)
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            await websocket.send_json({"text": transcribed_text, "error": None})
        except Exception as e:
            await websocket.send_json({"text": "", "error": f"Transcription failed: {str(e)}"})
    else:
        await websocket.send_json({"text": "", "error": "No audio data received."})
        
    await websocket.close()

        
    
