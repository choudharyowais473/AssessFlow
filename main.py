import os
#Thread setup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
#necessary imports that we require i am importing them once at start
import numpy as np
import logging
from slowapi import _rate_limit_exceeded_handler,Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import bcrypt
import torch
import sqlite3
import json
import openai
from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query ,Depends , HTTPException , status,Request
from pydantic import BaseModel
import requests
import asyncio
from cryptography.fernet import Fernet
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
from slowapi.middleware import SlowAPIMiddleware
#enviornment variables
load_dotenv(dotenv_path="Enviornment_Variable.env")
secret_key = os.getenv("Server_key")
API_KEY = os.getenv("API_KEY")
#pytorch thread optimization because it should work in low end pc
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to the specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)

oauth2_scheme=OAuth2PasswordBearer(tokenUrl="login")
model_sbert=SentenceTransformer("all-MiniLM-L6-v2")
model = WhisperModel("tiny.en", device="cpu", compute_type="float32", cpu_threads=2)
stop_words_set = set(stopwords.words('english'))
logging.basicConfig(filename='Uwais.log',filemode='w',format='%(asctime)s - %(levelname)s - %(message)s',level=logging.INFO)
import base64
import hashlib
from cryptography.fernet import Fernet, InvalidToken
fernet_key = base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode()).digest())
f = Fernet(fernet_key)
#defining all the basemodels
class signin(BaseModel):
    username:str
    password:str

class question_fetch(BaseModel):
    Field: str
    Difficulty: str

class scorepayload(BaseModel):
    answer: str
    explanation: str = ""
    full_explanation: List[str] | None = None
    questions: List[str] | None = None
    difficulty: List[str] | None = None
    current_difficulty: str
    previous_question: Optional[str] = ""
    session_token: str = ""

class finalscore(BaseModel):
    totalinfo:dict

class loginmodel(BaseModel):
    username:str
    password: str

class wiki(BaseModel):
    word_count:int
    session_token:str
#helper functions
def checkpassword(entered_pass: str, fetched_hash: bytes) -> bool:
    entered_bytes = entered_pass.encode('utf-8')
    return bcrypt.checkpw(entered_bytes, fetched_hash)

def encryptpass(entered_pass: str) -> bytes:
    hash_it = entered_pass.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(hash_it, salt)

async def similarity(answers: List[str], explanations: List[str]):
    answer_vecs = await asyncio.to_thread(model_sbert.encode, answers)
    explanation_vecs = await asyncio.to_thread(model_sbert.encode, explanations)
    similarity_scores = cosine_similarity(answer_vecs, explanation_vecs)
    return {"similarity": similarity_scores.diagonal().tolist()}

#Database intializing
logging.info("data base connecting")
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

#non endpoint main function
def get_similar_word_count(prev_question: str, current_question: str) -> tuple[int, float]:
    words_prev = set(word.lower() for word in prev_question.split() if word.lower() not in stop_words_set)
    words_current = set(word.lower() for word in current_question.split())
    
    similar_count = len(words_prev.intersection(words_current))
    overlap_ratio = similar_count / max(len(words_current), 1)
    
    return similar_count, overlap_ratio

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

import jwt
from fastapi import Request


def get_username_from_jwt(request: Request) -> str:

    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
       
        return request.client.host 
        
    try:
     
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, secret_key, algorithms='HS256')
        
        return payload.get("sub") or request.client.host
    except jwt.PyJWTError:
        
        return request.client.host

limiter=Limiter(key_func=get_username_from_jwt) 
app.state.limiter = limiter
import json
import openai
client = openai.OpenAI(base_url="http://localhost:3000/v1", api_key="local")

def generate_shortage_question(candidate_answers: list, candidate_scores: list, question_level: str) -> dict:
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
            max_tokens=150
        )
        
        raw_output = response.choices[0].message.content.strip()
        return json.loads(raw_output)
        
    except json.JSONDecodeError as e:
        print(f"Server AI JSON Error: {str(e)}")
        return {
            "question": "What is the primary purpose of a Docker multi-stage build?",
            "explanation": "Multi-stage builds optimize Dockerfile readability and size by separating build and runtime environments."
        }
    except Exception as e:
        print(f"Server AI Error: {str(e)}")
        return {"error": "Could not generate interview question template."}
    
# endpoints 
@app.post("/fetch_title")
@limiter.limit("5/minute")
async def title_fetch(request: Request, payload:wiki):
    logging.info("title_fetch initialized")
    explanation=payload.session_token.encode('utf-8')
    explanation=f.decrypt(explanation).decode('utf-8')
    logging.info(f"Fetching wiki with word_count: {payload.word_count}")
    url = "https://en.wikipedia.org/w/api.php"
    titles = []
    summaries=[]
    for i in range(len(explanation)):
    # Corrected parameters using a generator to enable 1200-character extracts
        params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",          # Flattens the response into clean lists
        "generator": "search",         # Replaces 'list=search' so we can get content
        "gsrsearch": explanation,     # Your search phrase (prefixed with 'gsr' for generators)
        "gsrlimit": 10,                # Number of search results to return
        "prop": "extracts",            # Activates text fetching
        "explaintext": True,           # Strips HTML tags for clean plain text
        "exchars": payload.word_count,               # Sets your custom maximum length
        "origin": "*"
    }
    
    headers = {
        "User-Agent": "SemanticSearchBot/1.0 (contact@example.com)"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        logging.info("Successfully fetched from wiki")
    except Exception as e :
        logging.error(f"Error fetching from wiki: {e}")
        params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",          # Flattens the response into clean lists
        "generator": "search",         # Replaces 'list=search' so we can get content
        "gsrsearch": explanation,     # Your search phrase (prefixed with 'gsr' for generators)
        "gsrlimit": 10,                # Number of search results to return
        "prop": "extracts",            # Activates text fetching
        "explaintext": True,           # Strips HTML tags for clean plain text
        "exchars": 500,               # Sets your custom maximum length
        "origin": "*"
    }
        headers = {
        "User-Agent": "SemanticSearchBot/1.0 (contact@example.com)"
    }
        try:
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            logging.info("Successfully fetched from wiki with fallback params")
        except Exception as e:
            logging.error(f"Fallback fetch failed: {e}")
            return {"error": str(e)}
    
    pages = data.get("query", {}).get("pages", [])
    
    for page in pages:
        title = page.get("title")
        summary = page.get("extract", "No summary available.")
        titles.append(title)
        summaries.append(summary)
        print(f"Title: {title}")
        print(f"Summary (Up to 1200 chars):\n{summary}") # i am sorry. but cleary out title and summary by yourself front end dev 
        print("-" * 60)
        
    return titles,summaries

@app.post("/signin")
async def signing(payload: signin):
    logging.info("signin initialized")
    username = payload.username
    password = payload.password
    conn = sqlite3.connect("Backend.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT username FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        if row is not None:
            return {"status": "user_already_exist"}
            
        cursor.execute(
            """INSERT INTO users (username, password, question_level, score) VALUES (?, ?, ?, ?)""",
            (username, encryptpass(password).decode('utf-8'), 0, 0.0)
        )
        conn.commit()
        logging.info(f"{username} signin successful")
        return {"status": "ID created"}
    except Exception as e:
        logging.error(f"Signin error: {e}")
        return {"status": "Error"}
    finally:
        conn.close()
    
@app.post("/login")
async def login(payload: loginmodel):
    logging.info(f'{payload.username} intialized login')
    username = payload.username
    password = payload.password
    conn = sqlite3.connect("Backend.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        logging.warning("no username found")
        return {"status": "Invalid"}
    user_db_id, stored_hash = row[0], row[1]
    if not checkpassword(password, stored_hash.encode('utf-8')):
        return {"status": "Invalid"}
    expire = datetime.utcnow() + timedelta(minutes=15)
    token_data = {"sub": username, "user_id": user_db_id, "exp": expire}
    access_token = jwt.encode(token_data, secret_key, algorithm="HS256")
    return {"status": "Valid", "access_token": access_token, "token_type": "bearer"}

@app.post("/question_fetch")
@limiter.limit("5/minute")
def question_fetcher(request: Request, question: question_fetch, username: str = None):
    logging.info("question fetching initialized")
    questions_info = question.model_dump()
    query = questions_info["Field"].strip('"').lower()
    difficulty_level = questions_info["Difficulty"].upper()
    
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
            token_payload = {"username": username, "explanations": explanations_level} if username else explanations_level
            session_token_bytes = f.encrypt(json.dumps(token_payload).encode('utf-8'))
            session_token = session_token_bytes.decode('utf-8')
            return {
                "questions": questions_level,  
                "difficulty": difficult_level,
                "session_token": session_token
            }
        except (TypeError, IndexError) as e:
            logging.error(f'{e}')
            return {"Error": str(e)}
    except requests.exceptions.RequestException as excp:
        logging.error(f'{e}')
        return {"excp": str(excp)}

@app.post('/start_interview')
def start_session(request: Request, payload: question_fetch, token:dict=Depends(user_info)):
    return question_fetcher(request, payload, token.get("sub"))

@app.post('/score')
async def score(payload: scorepayload,current_user:dict=Depends(user_info)):
    result_list = []
    
    if payload.session_token:
        token_bytes = payload.session_token.encode('utf-8') 
        try:
            decrypted_str = f.decrypt(token_bytes).decode('utf-8')
            token_data = json.loads(decrypted_str)
            if isinstance(token_data, dict) and "username" in token_data:
                if token_data["username"] != current_user["sub"]:
                    raise HTTPException(status_code=403, detail="Session token does not belong to this user")
                explanations_list = token_data.get("explanations", [])
            else:
                explanations_list = token_data if isinstance(token_data, list) else []
        except InvalidToken:
            raise HTTPException(status_code=400, detail="Invalid or tampered session token")
        except Exception:
            explanations_list = []
    else:
        explanations_list = payload.full_explanation if payload.full_explanation else []

    if not explanations_list:
        raise HTTPException(status_code=400, detail="Session expired or token already used.")

    current_exp = explanations_list[0] if explanations_list else payload.explanation
    
    sim_data = (await similarity([payload.answer], [current_exp])).get("similarity")
    result_list.append(sim_data)
    username=current_user["sub"]
    user_id=current_user["user_id"]
    final_score = float(sim_data[0])
    calculated_percentage = final_score * 100 
    
    question = payload.questions if payload.questions else []
    
    if explanations_list and len(explanations_list) > 0:
        explanations_list.pop(0)
        
    explanation = explanations_list
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
                    similar_count, overlap_ratio = get_similar_word_count(payload.previous_question, question[index])
                    
                    if overlap_ratio > 0.30 and len(difficulty) > 1:
                        continue
                    # ill use an local llm for this edge case soon
                    question_to_return = question.pop(index)
                    explanation_to_return = explanation.pop(index)
                    difficulty_to_return = difficulty.pop(index)
                    found_question = True
                    break
                except ValueError as e:
                    logging.error(f"{e} during scoring")
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
        forfrontend = generate_shortage_question(question, explanation, next_tier)
        question_to_return = forfrontend.get("question") or "What is the primary purpose of a Docker multi-stage build?"
        explanation_to_return = forfrontend.get("explanation") or "Multi-stage builds optimize Dockerfile readability and size by separating build and runtime environments."
        difficulty_to_return = next_tier
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
        logging.info("autorization and audio intialization")
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        username = payload.get("sub")
        if username is None:
            await websocket.close(code=1008, reason="Invalid token")
            return
    except PyJWTError as e:
        logging.error(f"{e}")
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
                data = await asyncio.wait_for(websocket.receive_bytes())
                sentence += data
                if len(sentence) >48000:
                    audio_array = np.frombuffer(sentence, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = await asyncio.to_thread(model.transcribe, audio_array, beam_size=3)
                    transcribed_text = " ".join([segment.text for segment in segments]).strip()
                    
                    await websocket.send_json({"text": transcribed_text, "error": None,"type":"Partial"})
            except Exception as e:
                logging.info(f"{e}")
                await websocket.send_json({"text": "", "error": f"Transcription failed: {str(e)}"})
            else:
                await websocket.send_json({"text": "", "error": "No audio data received."})
                
    except WebSocketDisconnect:
        await websocket.close()
        
    if len(sentence) > 0:
        try:
            audio_array = np.frombuffer(sentence, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = await asyncio.to_thread(model.transcribe, audio_array, beam_size=3)
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            await websocket.send_json({"text": transcribed_text, "error": None,"type":"Partial"})
        except Exception as e:
            await websocket.send_json({"text": "", "error": f"Transcription failed: {str(e)}","type":"Partial"})
    else:
        await websocket.send_json({"text": "", "error": "No audio data received.","type":"Partial"})
        
    await websocket.close()

 # i almost completed this code yeah maybe some error handling remains but i guess well first test it and find out then ill update it 

@app.post("/score/finalinfo")
def finalscores(payload: finalscore,token:dict=Depends(user_info)):
    totalinfo = payload.totalinfo
    username = token.get("sub")
    score = totalinfo.get("score", 0.0)
    # Ensure score is between 0.0 and 1.0 for the DB constraint
    if score > 1.0:
        score = score / 100.0
    score = max(0.0, min(1.0, float(score)))
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
        logging.error("e")
        raise HTTPException(status_code=500, detail="Database write failure")
    finally:
        conn.close()
    return {"status": "successful"}
    
