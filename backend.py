
import os
from typing import List
from fastapi import FastAPI
import asyncio
from ultralytics import YOLO
from pytimer2 import Timer # this is a third party easy to implement library thats why i choose it lol

import requests
from vosk import Model, KaldiRecognizer
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import joblib as jb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
app=FastAPI() 
model=Model("vosk-model-small-en-us-0.15")
'''
def vision_model(voilation):
    keep_streaming=True
    yolo_frameskip=20
    model=YOLO("yolo11n.pt")
    frame_count=0
    timer=Timer()
    cheating_dict={}
    time_running=False
    active_violation=None
    mesh_model=mp.solutions.face_mesh.FaceMesh
    face_mesh=mesh_model(max_num_faces=1,
                         refine_landmarks=True,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5)
    left_eye_R=33
    left_eye_L=133
    left_pupil=468
    right_eye_R=362
    right_eye_L=263
    right_pupil=473
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    new_status=0
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return {"Frame":None}

    print("Webcam started! Waiting for stream synchronization...")

    print("Live stream active. Press 'q' on your keyboard to close the window.")
    start=time.time()
    gaze_direction = "No Face Detected"
    while keep_streaming:
            gaze_direction = "No Face Detected"
            end_time=time.time()
            s=end_time-start
            frame_count+=1
            try:
                fps=frame_count/s # i am adding this just in case if anyone wnts to see frame persecond
            except ZeroDivisionError:
                fps=frame_count/s+1
            ret,frame=cap.read()
            if not ret or frame is None:
                continue
            frame=cv.flip(frame,1)
            rgb_image=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            results=face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_right=face_landmarks.landmark[left_eye_R]
                    left_eye_left=face_landmarks.landmark[left_eye_L]
                    left_pupil_pos=face_landmarks.landmark[left_pupil]
                    right_eye_right=face_landmarks.landmark[right_eye_R]
                    right_eye_left=face_landmarks.landmark[right_eye_L]
                    right_pupil_pos=face_landmarks.landmark[right_pupil]
                    h,w,c=frame.shape
                    right_eye_right_x=int(right_eye_right.x*w)
                    right_eye_right_y=int(right_eye_right.y*h)
                    right_eye_left_x=int(right_eye_left.x*w)
                    right_eye_left_y=int(right_eye_left.y*h)
                    right_pupil_x=int(right_pupil_pos.x*w)
                    right_pupil_y=int(right_pupil_pos.y*h)
                    left_eye_right_x=int(left_eye_right.x*w)
                    left_eye_right_y=int(left_eye_right.y*h)
                    left_eye_left_x=int(left_eye_left.x*w)
                    left_eye_left_y=int(left_eye_left.y*h)
                    left_pupil_x=int(left_pupil_pos.x*w)
                    left_pupil_y=int(left_pupil_pos.y*h)
                    left_ratio_x=(left_eye_left_x-left_pupil_x)/max(1,abs(left_eye_left_x-left_eye_right_x))
                    left_ratio_y=(left_eye_left_y-left_pupil_y)/max(1,abs(left_eye_left_y-left_eye_right_y))
                    right_ratio_x=(right_eye_left_x-right_pupil_x)/max(1,abs(right_eye_left_x-right_eye_right_x))
                    right_ratio_y=(right_eye_left_y-right_pupil_y)/max(1,abs(right_eye_left_y-right_eye_right_y))
                    gaze_threshold=0.35
                    if (gaze_threshold <= left_ratio_x <= (1 - gaze_threshold)) and \
                       (gaze_threshold <= right_ratio_x <= (1 - gaze_threshold)):
                        gaze_direction="Looking Center"
                    elif left_ratio_x < gaze_threshold or right_ratio_x < gaze_threshold:
                        gaze_direction = "Looking Right"
                    elif left_ratio_x > (1 - gaze_threshold) or right_ratio_x > (1 - gaze_threshold):
                        gaze_direction = "Looking Left"
                    elif left_ratio_y < gaze_threshold or right_ratio_y < gaze_threshold:
                        gaze_direction = "Looking Down"
                        timer.start_countdown(duration=5)
                    elif left_ratio_y > (1 - gaze_threshold) or right_ratio_y > (1 - gaze_threshold):
                        gaze_direction = "Looking Up"
                    else:
                        gaze_direction = "Looking Center"
            if gaze_direction!="Looking Center":
                if not time_running:
                    timer.start_countdown(duration=5)
                    active_violation=gaze_direction
                    time_running=True   
                if timer.get_countdown()<=0:
                        cheating_dict[active_violation] = cheating_dict.get(active_violation, 0) + 1
                        voilation.put(active_violation)
                        time_running=False
                        active_violation=None
                        
            else:
                if time_running:
                    time_running=False
                    active_violation=None
            if frame_count%20==0: # i set it at 20 because my pc was giving about 26 fps so its basically checking 1 times per second anyways you can adjust it yourself
                logging.getLogger("ultralytics").setLevel(logging.WARNING)# to hide unnecessary ugly warnings if yolo gave any. if its not working on youyr system just remove this line amd run the code for debugging
                results=model(frame,classes=[67,63],imgsz=320)
                for result in results:
                    for box in result.boxes:
                        keep_streaming=False
                        break

    cap.release()
    cv.destroyAllWindows()
    print("Cheating Instances Detected:", cheating_dict)
    return {"viloation":voilation}'''
class question_fetch(BaseModel):
    Field: str
    Difficulty:str
    API_KEY: str
class scorepayload(BaseModel):
    answer:str
    explanation:str
def title_fetch(query_phrase:str):
    url= "https://en.wikipedia.org/w/api.php"
    params={
        "action": "query",
        "list": "search",
        "srsearch": query_phrase,        
        "format": "json",
        "origin": "*"
    }
    headers = {
        "User-Agent": "SemanticSearchBot/1.0 (contact@example.com)"
    }
    response=requests.get(url,params=params,headers=headers)
    data=response.json()
    search_results = data.get("query", {}).get("search", [])
    titles=[]
    for i in range(len(search_results)):
        data_point=search_results[i].get("title")
        titles.append(data_point)
    print(titles)
    return {"titles":titles}
@app.websocket("/STT")
async def speech_to_text(websocket: WebSocket):
    fs = 16000 
    rec=KaldiRecognizer(model,fs)
    sentence=[]
    '''p=pyaudio.PyAudio() # had to comment it all out because frontend plan changed
    sentence_q=queue.Queue()  
    fs = 16000   
    f = 440.0        
    Chunk = 8000
    model=Model("vosk-model-small-en-us-0.15")
    rec=KaldiRecognizer(model,fs)
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=fs,input=True,frames_per_buffer=Chunk)
    stream.start_stream()
    start=time.time()
    print("Starting audio")
    sentence=[]
    partials=[]'''
    await websocket.accept()
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(),timeout=5.0)
                if rec.AcceptWaveform(data):
                    result=json.loads(rec.Result())
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
def text_to_speech(text_to_say:str):
    safe_text = str(text_to_say).replace('"', '\\"')
    word_count = len(safe_text.split())
    estimated_duration = int((word_count / 2.5) + 1.0)
    
    print(f"Computer Speaking: {text_to_say}")
    os.system(f'say "{safe_text}" && sleep {estimated_duration} &')
def similarity(answers:List[str],explanations:List[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    answer_embeddings = model.encode(answers)
    explanation_embeddings = model.encode(explanations) 
    similarity_scores = cosine_similarity(answer_embeddings, explanation_embeddings)
    print(similarity_scores.diagonal().tolist())
    return {"similarity":similarity_scores.diagonal().tolist() } 
@app.post("/question_fetch")
def question_fetcher(question: question_fetch):
    questions_info=question.model_dump()
    query=questions_info["Field"]
    difficulty_level=questions_info["Difficulty"]
    API_KEY= questions_info["API_KEY"]
    url="https://quizapi.io/api/v1/questions"
    query=query.strip('"')
    query=query.lower()
    difficulty_level=difficulty_level.upper()
    params={
        "category": query,
        "difficulty":difficulty_level,
        "limit":20,
        "offset":0
    }
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response=requests.get(url=url,params=params,headers=headers,timeout=5)
        data=response.json()
        data=data.get('data')
        sample_size=min(5,len(data))
        questions=[]
        explanations=[]
        numbers=random.sample(range(0, len(data)), sample_size)     
        for index, random_pos in enumerate(numbers):         
            ques=data[random_pos].get('text')  
            explanation=data[random_pos].get('explanation')
            questions.append(ques)  
            explanations.append(explanation) 
            print(f"{index + 1}. {ques}\n")
            print(f"{index+1}.{explanation}\n")
        return {"questions":questions,"explanation":explanations}
    except requests.exceptions.RequestException as excp:
        return {"excp":excp}
def assessment(local_db):
    data=pd.read_sql("SELECT level,anamoly,Answer_similarity FROM assessment_table",local_db)
    model=KMeans(n_clusters=3,random_state=42)
    ct=ColumnTransformer(transformers=[("ordenc",OrdinalEncoder(),["anamoly","level"])])
    pipeline=Pipeline([('ct',ct),('kmeans',model
    )])
    data['Clusters']=pipeline.fit_predict(data)
    X_transformed=pipeline.named_steps['ct'].transform(data)
    distances=pipeline.named_steps['kmeans'].transform(X_transformed)
    data['Score']=[distances[i,c] for i,c in enumerate(data['Clusters'])]
    cluster_id=data.groupby('Clusters')['Answer_similarity'].mean().sort_values(ascending=False)
    mapping={
        cluster_id.index[0]:"Top Performer",
        cluster_id.index[1]:"Averge Performer",
        cluster_id.index[2]:"Poor Performer"
    }
    data['Performance_Group']=data['Clusters'].map(mapping)
    siliscore=silhouette_score(X_transformed,data['Clusters'])
    print(f"Silhouette Score:{siliscore}")
    jb.dump(pipeline,"kmeans_model.joblib")
@app.post('/start_interview')
def start_session(payload:question_fetch):
    result=question_fetcher(payload)
    return result
@app.post('/score')
def score(payload:scorepayload):
    result=similarity([payload.answer],[payload.explanation])
    return result
@app.get("/hello")
def hello():
    return {"hi":"hello"}