import numpy as np
import os
import bcrypt
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
import sqlite3
import json
import openai
from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query ,Depends , HTTPException , status
from pydantic import BaseModel
import requests
import asyncio
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import random
from faster_whisper import WhisperModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from datetime import datetime,timedelta
import jwt
from jwt import PyJWTError
from dotenv import load_dotenv
load_dotenv(dotenv_path="Enviornment_Variable.env")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
app = FastAPI()
oauth2_scheme=OAuth2PasswordBearer(tokenUrl="login")
model_sbert=SentenceTransformer("all-MiniLM-L6-v2")
model = WhisperModel("tiny.en", device="cpu", compute_type="float32", cpu_threads=2)
stop_words_set = set(stopwords.words('english'))
def checkpassword(entered_pass: str, fetched_hash: bytes) -> bool:
    entered_bytes = entered_pass.encode('utf-8')
    return bcrypt.checkpw(entered_bytes, fetched_hash)

def encryptpass(entered_pass: str) -> bytes:
    hash_it = entered_pass.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(hash_it, salt)

conn = sqlite3.connect("Backend.db")
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    question_level INTEGER DEFAULT 0,
    score REAL DEFAULT 0.0 CHECK (score >= 0.0 AND score <= 1.0)
);
""")
cursor.execute("SELECT COUNT(*) FROM users")
if cursor.fetchone()[0] == 0:
    h_mid = encryptpass("hash1").decode('utf-8')
    h_high = encryptpass("hash2").decode('utf-8')
    h_new = encryptpass("hash3").decode('utf-8')
    h_test = encryptpass("4532pqsdfv@kl").decode('utf-8')
    
    cursor.executemany("""
    INSERT INTO users (username, password, question_level, score) 
    VALUES (?, ?, ?, ?);
    """, [
        ('player_mid', h_mid, 4, 0.5),
        ('player_high', h_high, 7, 0.85),
        ('player_new', h_new, 0, 0.0),
        ('testuser', h_test, 0, 0.0)
    ])
    conn.commit()
conn.close()
class question_fetch(BaseModel):
    Field: str
    Difficulty: str

class scorepayload(BaseModel):
    answer: str
    explanation: str
    full_explanation: List[str]
    questions: List[str] | None = None
    difficulty: List[str] | None = None
    current_difficulty: str
    previous_question: Optional[str] = ""

class finalscore(BaseModel):
    totalinfo:dict
class loginmodel(BaseModel):
    username:str
    password: str
def get_similar_word_count(prev_question: str, current_question: str) -> tuple[int, float]:
    words_prev = set(word.lower() for word in prev_question.split() if word.lower() not in stop_words_set)
    words_current = set(word.lower() for word in current_question.split())
    
    similar_count = len(words_prev.intersection(words_current))
    overlap_ratio = similar_count / max(len(words_current), 1)
    
    return similar_count, overlap_ratio
secret_key = os.getenv("Server_key")

@app.post("/login")
async def login(payload: loginmodel):
    username = payload.username
    password = payload.password
    conn = sqlite3.connect("Backend.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return {"status": "Invalid"}
    user_db_id, stored_hash = row[0], row[1]
    if not checkpassword(password, stored_hash.encode('utf-8')):
        return {"status": "Invalid"}
    expire = datetime.utcnow() + timedelta(minutes=15)
    token_data = {"sub": username, "user_id": user_db_id, "exp": expire}
    access_token = jwt.encode(token_data, secret_key, algorithm="HS256")
    return {"status": "Valid", "access_token": access_token, "token_type": "bearer"}

def user_info(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        conn = sqlite3.connect("Backend.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        conn.close()
        if username is None or row is None:
            raise credentials_exception
        return {"sub": username, "user_id": row[0]}
    except PyJWTError:
        raise credentials_exception
    
import json
import openai
client = openai.OpenAI(base_url="http://localhost:3000/v1", api_key="local")
LLM_AVAILABLE = True
try:
    client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        timeout=2
    )
except Exception:
    LLM_AVAILABLE = False
    print("No local LLM detected at startup — fallback question generation disabled, using static fallback instead.")

DEFAULT_FALLBACK_QUESTIONS = {
    "EASY": {
        "question": "What is a variable in programming?",
        "explanation": "A variable is a named storage location used to hold a value that can change during program execution."
    },
    "MEDIUM": {
        "question": "What is the difference between a list and a tuple in Python?",
        "explanation": "Lists are mutable and can be changed after creation, while tuples are immutable and cannot be modified once created."
    },
    "HARD": {
        "question": "What is the primary purpose of a Docker multi-stage build?",
        "explanation": "Multi-stage builds optimize Dockerfile readability and size by separating build and runtime environments."
    }
}

def generate_shortage_question(candidate_answers: list, candidate_scores: list, question_level: str) -> dict:
    fallback = DEFAULT_FALLBACK_QUESTIONS.get(question_level, DEFAULT_FALLBACK_QUESTIONS["MEDIUM"])

    if not LLM_AVAILABLE:
        return fallback

    prompt_content = (
        f"Generate a new interview question of {question_level} difficulty. "
        f"The candidate's previous answers were {candidate_answers} with scores {candidate_scores}. "
        f"Provide a relevant question and a one-sentence explanation. "
        f"You MUST output a JSON object with exactly the keys 'question' and 'explanation'."
    )
    
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a strict data-serialization server. Output ONLY raw, unformatted JSON. Do not include markdown code blocks."
                },
                {"role": "user", "content": prompt_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.2, 
            max_tokens=150,
            timeout=5
        )
        
        raw_output = response.choices[0].message.content.strip()
        parsed = json.loads(raw_output)
        if "question" not in parsed or "explanation" not in parsed:
            return fallback
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"Server AI JSON Error: {str(e)}")
        return fallback
    except Exception as e:
        print(f"Server AI Error: {str(e)}")
        return fallback
    

@app.post("/question_fetch")
def question_fetcher(question: question_fetch):
    questions_info = question.model_dump()
    query = questions_info["Field"].strip('"').lower()
    difficulty_level = questions_info["Difficulty"].upper()
    API_KEY = os.getenv("API_KEY")
    
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
async def score(payload: scorepayload,current_user:dict=Depends(user_info)):
    result_list = []
    sim_data = (await similarity([payload.answer], [payload.explanation])).get("similarity")
    result_list.append(sim_data)
    username=current_user["sub"]
    user_id=current_user["user_id"]
    final_score = float(sim_data[0])
    calculated_percentage = final_score * 100 
    
    question = payload.questions if payload.questions else []
    explanation = payload.full_explanation if payload.full_explanation else []
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
    next_tier = payload.current_difficulty
    
    PASS_THRESHOLD = 65.0 
    
    if calculated_percentage >= PASS_THRESHOLD:
        next_tier = difficult_range[min(position + 1, len(difficult_range) - 1)]
        for index in range(len(difficulty) - 1, -1, -1):
            if difficulty[index] == next_tier:
                try:
                    similar_count, overlap_ratio = get_similar_word_count(payload.previous_question, question[index])
                    
                    if overlap_ratio > 0.30 and len(difficulty) > 1:
                        continue
                    # ill use an local llm for this edge case soon
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
                if payload.current_difficulty=="EASY":
                    return{"status":"No question in inventory"} 
                question_to_return=question.pop(index)
                explanation_to_return = explanation.pop(index)
                difficulty_to_return = difficulty.pop(index)
                found_question = True
                break
                
    if not found_question and question:
        fallback_level = next_tier
        forfrontend = generate_shortage_question(question, explanation, fallback_level)
        fallback_default = DEFAULT_FALLBACK_QUESTIONS.get(fallback_level, DEFAULT_FALLBACK_QUESTIONS["MEDIUM"])
        question_to_return = forfrontend.get("question") or fallback_default["question"]
        explanation_to_return = forfrontend.get("explanation") or fallback_default["explanation"]
        difficulty_to_return = fallback_level
    # Before returning use user id and username to add this to database of the user id for now since db is not my role i am putting this empty
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
async def get_voice(websocket: WebSocket, mic_status: bool = Query(True),token:str=Query("")):
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        username = payload.get("sub")
        if username is None:
            await websocket.close(code=1008, reason="Invalid token")
            return
    except PyJWTError:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
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

 # i almost completed this code yeah maybe some error handling remains but i guess well first test it and find out then ill update it 
@app.post("/finalinfo")
def finalscores(payload: finalscore, token: dict = Depends(user_info)):
    totalinfo = payload.totalinfo
    username = totalinfo.get("username")
    score = totalinfo.get("score")
    question_level = totalinfo.get("question_lvl")
    
    conn = sqlite3.connect("Backend.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        if row:
            cursor.execute(
                "UPDATE users SET question_level=?, score=? WHERE username=?",
                (question_level, score, username)
            )
        else:
            dummy_hash = encryptpass("password").decode('utf-8')
            cursor.execute(
                "INSERT INTO users (username, password, question_level, score) VALUES (?, ?, ?, ?)",
                (username, dummy_hash, question_level, score)
            )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error in finalinfo: {e}")
        raise HTTPException(status_code=500, detail="Database write failure")
    finally:
        conn.close()
    return {"status": "successful"}
    
