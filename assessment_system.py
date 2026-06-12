from multiprocessing import Process, Queue
import multiprocessing
import os
import queue
import cv2 as cv
from ultralytics import YOLO
from pytimer2 import Timer
import mediapipe as mp
import time
import logging
import requests
import pyaudio
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


def vision_model(violation):
    """
    Monitor student gaze direction during assessment.
    Detects looking away and phone/person presence.
    """
    keep_streaming = True
    model = YOLO("yolo11n.pt")
    frame_count = 0
    timer = Timer()
    cheating_dict = {}
    time_running = False
    active_violation = None
    
    mesh_model = mp.solutions.face_mesh.FaceMesh
    face_mesh = mesh_model(max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    
    left_eye_R = 33
    left_eye_L = 133
    left_pupil = 468
    right_eye_R = 362
    right_eye_L = 263
    right_pupil = 473
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return {"violations": cheating_dict}

    print("Webcam started! Waiting for stream synchronization...")
    print("Live stream active. Press 'q' on your keyboard to close the window.")
    
    start = time.time()
    gaze_direction = "No Face Detected"
    
    while keep_streaming:
        gaze_direction = "No Face Detected"
        end_time = time.time()
        s = end_time - start
        frame_count += 1
        
        try:
            fps = frame_count / s
        except ZeroDivisionError:
            fps = 0
        
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        frame = cv.flip(frame, 1)
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        # Gaze direction detection
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_right = face_landmarks.landmark[left_eye_R]
                left_eye_left = face_landmarks.landmark[left_eye_L]
                left_pupil_pos = face_landmarks.landmark[left_pupil]
                right_eye_right = face_landmarks.landmark[right_eye_R]
                right_eye_left = face_landmarks.landmark[right_eye_L]
                right_pupil_pos = face_landmarks.landmark[right_pupil]
                
                h, w, c = frame.shape
                right_eye_right_x = int(right_eye_right.x * w)
                right_eye_right_y = int(right_eye_right.y * h)
                right_eye_left_x = int(right_eye_left.x * w)
                right_eye_left_y = int(right_eye_left.y * h)
                right_pupil_x = int(right_pupil_pos.x * w)
                right_pupil_y = int(right_pupil_pos.y * h)
                left_eye_right_x = int(left_eye_right.x * w)
                left_eye_right_y = int(left_eye_right.y * h)
                left_eye_left_x = int(left_eye_left.x * w)
                left_eye_left_y = int(left_eye_left.y * h)
                left_pupil_x = int(left_pupil_pos.x * w)
                left_pupil_y = int(left_pupil_pos.y * h)
                
                left_ratio_x = (left_eye_left_x - left_pupil_x) / max(1, abs(left_eye_left_x - left_eye_right_x))
                left_ratio_y = (left_eye_left_y - left_pupil_y) / max(1, abs(left_eye_left_y - left_eye_right_y))
                right_ratio_x = (right_eye_left_x - right_pupil_x) / max(1, abs(right_eye_left_x - right_eye_right_x))
                right_ratio_y = (right_eye_left_y - right_pupil_y) / max(1, abs(right_eye_left_y - right_eye_right_y))
                
                gaze_threshold = 0.35
                
                if (gaze_threshold <= left_ratio_x <= (1 - gaze_threshold)) and \
                   (gaze_threshold <= right_ratio_x <= (1 - gaze_threshold)):
                    gaze_direction = "Looking Center"
                elif left_ratio_x < gaze_threshold or right_ratio_x < gaze_threshold:
                    gaze_direction = "Looking Right"
                elif left_ratio_x > (1 - gaze_threshold) or right_ratio_x > (1 - gaze_threshold):
                    gaze_direction = "Looking Left"
                elif left_ratio_y < gaze_threshold or right_ratio_y < gaze_threshold:
                    gaze_direction = "Looking Down"
                elif left_ratio_y > (1 - gaze_threshold) or right_ratio_y > (1 - gaze_threshold):
                    gaze_direction = "Looking Up"
                else:
                    gaze_direction = "Looking Center"
        
        # Violation tracking
        if gaze_direction != "Looking Center":
            if not time_running:
                timer.start_countdown(duration=5)
                active_violation = gaze_direction
                time_running = True
            
            if timer.get_countdown() <= 0:
                cheating_dict[active_violation] = cheating_dict.get(active_violation, 0) + 1
                violation.put(active_violation)
                time_running = False
                active_violation = None
        else:
            if time_running:
                time_running = False
                active_violation = None
        
        # YOLO detection every 8 frames
        if frame_count % 8 == 0:
            logging.getLogger("ultralytics").setLevel(logging.WARNING)
            results = model(frame, classes=[67, 63], imgsz=320)
            for result in results:
                for box in result.boxes:
                    keep_streaming = False
                    break
        
        # Stop after 30 seconds
        if s > 30:
            keep_streaming = False
            break

    cap.release()
    cv.destroyAllWindows()
    print("Cheating Instances Detected:", cheating_dict)
    return {"violations": cheating_dict}


def title_fetch(query_phrase: str):
    """
    Fetch Wikipedia titles matching a search query.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query_phrase,
        "format": "json",
        "origin": "*"
    }
    headers = {
        "User-Agent": "SemanticSearchBot/1.0 (contact@example.com)"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        titles = [result.get("title") for result in search_results]
        print(f"Found titles: {titles}")
        return {"titles": titles}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching titles: {e}")
        return {"titles": [], "error": str(e)}


def speech_to_text():
    """
    Capture and transcribe speech using Vosk.
    Stops after 5 seconds of silence.
    """
    p = pyaudio.PyAudio()
    fs = 16000
    chunk = 8000
    
    model = Model("vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, fs)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, 
                    input=True, frames_per_buffer=chunk)
    stream.start_stream()
    
    start = time.time()
    print("Starting audio capture...")
    sentence = []
    partials = []
    
    while True:
        end = time.time()
        elapsed = end - start
        
        if stream.get_read_available() < chunk:
            continue
        
        if elapsed >= 5:
            print("5 seconds of silence reached. Exiting.")
            break
        
        data = stream.read(chunk, exception_on_overflow=False)
        
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result and result.get('text'):
                start = time.time()
                print(f"Recognized: {result['text']}")
                sentence.append(result['text'])
        else:
            partial = json.loads(rec.PartialResult())
            if partial.get('partial'):
                if partial not in partials:
                    print(f"Partial: {partial}")
                    partials.append(partial)
                    start = time.time()
                time.sleep(0.1)
    
    final_result = json.loads(rec.FinalResult())
    if final_result and final_result.get('text'):
        print(f"Final: {final_result['text']}")
        sentence.append(final_result['text'])
    
    final_sentence = ' . '.join(sentence)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return {"text": final_sentence}


def text_to_speech(text_to_say):
    """
    Convert text to speech using system 'say' command (macOS).
    Runs asynchronously in background.
    """
    safe_text = str(text_to_say).replace('"', '\\"')
    word_count = len(safe_text.split())
    estimated_duration = int((word_count / 2.5) + 1.0)
    
    print(f"Computer Speaking: {text_to_say}")
    os.system(f'say "{safe_text}" && sleep {estimated_duration} &')


def similarity(answers, explanations):
    """
    Calculate cosine similarity between student answers and explanations.
    Expects lists of strings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    answer_embeddings = model.encode(answers)
    explanation_embeddings = model.encode(explanations)
    similarity_scores = cosine_similarity(answer_embeddings, explanation_embeddings)
    scores = similarity_scores.diagonal().tolist()
    print(f"Similarity scores: {scores}")
    return {"similarity": scores}


def question_fetcher(query: str, difficulty_level: str, api_key: str):
    """
    Fetch random questions from QuizAPI.
    Returns up to 5 questions with explanations.
    """
    url = "https://quizapi.io/api/v1/questions"
    query = query.strip('"').lower()
    difficulty_level = difficulty_level.upper()
    
    params = {
        "category": query,
        "difficulty": difficulty_level,
        "limit": 20,
        "offset": 0
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        response = requests.get(url=url, params=params, headers=headers, timeout=5)
        data = response.json().get('data', [])
        
        if not data:
            return {"questions": [], "explanations": [], "error": "No questions found"}
        
        sample_size = min(5, len(data))
        questions = []
        explanations = []
        random_indices = random.sample(range(len(data)), sample_size)
        
        for index, random_pos in enumerate(random_indices):
            ques = data[random_pos].get('text', '')
            explanation = data[random_pos].get('explanation', '')
            questions.append(ques)
            explanations.append(explanation)
            print(f"{index + 1}. {ques}\n")
            print(f"Explanation: {explanation}\n")
        
        return {"questions": questions, "explanations": explanations}
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return {"questions": [], "explanations": [], "error": str(e)}


def assessment(local_db):
    """
    Perform KMeans clustering on assessment data and classify performance.
    Saves trained pipeline to disk.
    """
    try:
        data = pd.read_sql(
            "SELECT level, anamoly, Answer_similarity FROM assessment_table", 
            local_db
        )
        
        if data.empty:
            print("No assessment data found")
            return {"error": "No data"}
        
        model = KMeans(n_clusters=3, random_state=42)
        ct = ColumnTransformer(
            transformers=[("ordenc", OrdinalEncoder(), ["anamoly", "level"])],
            remainder='passthrough'
        )
        pipeline = Pipeline([('ct', ct), ('kmeans', model)])
        
        data['Clusters'] = pipeline.fit_predict(data)
        X_transformed = pipeline.named_steps['ct'].transform(data)
        distances = pipeline.named_steps['kmeans'].transform(X_transformed)
        
        data['Score'] = [distances[i, c] for i, c in enumerate(data['Clusters'])]
        
        cluster_performance = data.groupby('Clusters')['Answer_similarity'].mean().sort_values(ascending=False)
        mapping = {
            cluster_performance.index[0]: "Top Performer",
            cluster_performance.index[1]: "Average Performer",
            cluster_performance.index[2]: "Poor Performer"
        }
        data['Performance_Group'] = data['Clusters'].map(mapping)
        
        silhouette = silhouette_score(X_transformed, data['Clusters'])
        print(f"Silhouette Score: {silhouette}")
        
        jb.dump(pipeline, "kmeans_model.joblib")
        return {
            "silhouette_score": silhouette,
            "performance_groups": data['Performance_Group'].value_counts().to_dict()
        }
    
    except Exception as e:
        print(f"Error in assessment: {e}")
        return {"error": str(e)}


def run_assessment(api_key: str, category: str = "Physics", difficulty: str = "EASY"):
    """
    Main assessment workflow:
    1. Fetch questions
    2. For each question: monitor vision, capture audio, evaluate similarity
    3. Log cheating instances
    """
    multiprocessing.set_start_method('spawn', force=True)
    
    print("Fetching questions...")
    result = question_fetcher(category, difficulty, api_key)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    questions = result["questions"]
    explanations = result["explanations"]
    
    if not questions:
        print("No questions fetched")
        return
    
    all_results = []
    
    for i in range(len(questions)):
        print(f"\n--- Question {i + 1} ---")
        print(f"Q: {questions[i]}")
        
        # Vision monitoring
        violation = Queue()
        p1 = Process(target=vision_model, args=(violation,))
        p1.start()
        
        # Ask question
        text_to_speech(questions[i])
        time.sleep(5)
        
        # Capture response
        student_response = speech_to_text()
        
        # Stop vision monitoring
        p1.terminate()
        p1.join()
        
        # Collect violations
        cheating_logs = []
        while not violation.empty():
            try:
                cheating_logs.append(violation.get_nowait())
            except queue.Empty:
                break
        
        # Evaluate similarity
        sim_result = similarity(
            [student_response["text"]], 
            [explanations[i]]
        )
        
        result_entry = {
            "question_num": i + 1,
            "question": questions[i],
            "student_answer": student_response["text"],
            "similarity_score": sim_result["similarity"][0],
            "violations": cheating_logs
        }
        all_results.append(result_entry)
        
        print(f"Student Response: {student_response['text']}")
        print(f"Similarity: {sim_result['similarity'][0]:.3f}")
        print(f"Violations: {cheating_logs}")
    
    return all_results


if __name__ == '__main__':
    # Example usage
    API_KEY = "API_KEY"
    results = run_assessment(API_KEY)
    
    if results:
        print("\n=== ASSESSMENT SUMMARY ===")
        for r in results:
            print(f"Q{r['question_num']}: Similarity={r['similarity_score']:.3f}, Violations={len(r['violations'])}")
