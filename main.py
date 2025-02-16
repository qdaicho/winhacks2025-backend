import asyncio
import io
import os
import glob
import subprocess
import sqlite3
import numpy as np
import librosa
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header, Depends
from datetime import datetime
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import assemblyai as aai
import logging
import math
import time
import nltk
import textstat
import uuid
from pydub import AudioSegment
import speech_recognition as sr 
from pydub.silence import split_on_silence

# NLTK Setup
nltk.download("punkt")
from nltk.tokenize import word_tokenize

# Initialize AssemblyAI client (if needed)
aai.settings.api_key = "43d41404b4004459a42547709c069416"

import uvicorn

import torch
import torchaudio
from transformers import pipeline
from dataclasses import dataclass
from typing import List
from collections import Counter

# -------------------------------
# Create or Connect to SQLite
# -------------------------------
def init_db():
    """
    Initialize the SQLite database and create the transcriptions table if not exists.
    """
    with sqlite3.connect("transcriptions.db") as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                transcript TEXT,
                wpm_30 REAL,
                filler_count_30 INTEGER,
                filler_pct_30 REAL,
                readability_60 REAL,
                freq_60 TEXT,
                total_pauses INTEGER,
                global_word_count INTEGER,
                global_filler_count INTEGER,
                global_filler_pct REAL,
                global_readability REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

# Initialize DB
init_db()

# -------------------------------
# Constants & Globals
# -------------------------------
audio_buffer = b""  # Stores incoming WebM audio data (if needed)
SAMPLE_RATE = 16000
FRAME_SIZE = 2       
CHANNELS = 1         
CHUNK_DURATION = 1.0

connected_clients = set()  
OUTPUT_WEBM_FILE = "received_audio.webm"
OUTPUT_WAV_FILE  = "received_audio.wav"
audio_chunks = []           

@dataclass
class RecognizedWord:
    word: str
    timestamp: float  # Seconds from the start of the speech session

speech_history: List[RecognizedWord] = []
FILLER_WORDS = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally"}

# -------------------------------
# FastAPI Initialization
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Helper: Check Bearer vs. Path
# -------------------------------
def authorize_user_id(websocket: WebSocket, expected_user_id: str) -> None:
    """
    Confirms the Authorization header is 'Bearer <expected_user_id>'.
    Raises WebSocketDisconnect if invalid or mismatched.
    """
    auth_header = websocket.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise WebSocketDisconnect(
            code=4001,
            reason="Missing or invalid Authorization header."
        )
    token_user_id = auth_header[len("Bearer "):].strip()
    if token_user_id != expected_user_id:
        raise WebSocketDisconnect(
            code=4003,
            reason="Token userId does not match path userId."
        )

# -------------------------------
# Non-WebSocket Endpoints
# (unchanged from original)
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Speech Analysis WebSocket Server is up and running!"}

@app.get("/token", response_class=JSONResponse)
async def get_token(request: Request):
    """
    Example route to generate an AssemblyAI token (requires Bearer user_id).
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    user_id = auth_header[len("Bearer "):].strip()

    try:
        token = aai.RealtimeTranscriber.create_temporary_token(expires_in=3600)
        return JSONResponse(content={"token": token, "for_user": user_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hello")
async def hello_world(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    user_id = auth_header[len("Bearer "):].strip()
    return {"message": f"Hello, World! (from user: {user_id})"}

@app.get("/history/{userId}", response_class=JSONResponse)
async def get_user_history(userId: str, request: Request):
    """
    Return all transcriptions in JSON for the given userId.
    Expects 'Authorization: Bearer <userId>' to match the path userId.
    """
    # 1. Verify bearer token matches userId
    # auth_header = request.headers.get("Authorization")
    # if not auth_header or not auth_header.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    # token_user_id = auth_header[len("Bearer "):].strip()
    # if token_user_id != userId:
    #     raise HTTPException(status_code=403, detail="Token userId does not match path userId.")

    # 2. Query the database for all rows belonging to userId
    with sqlite3.connect("transcriptions.db") as conn:
        conn.row_factory = sqlite3.Row  # Allows us to get dict-like rows
        c = conn.cursor()
        c.execute("SELECT * FROM transcriptions WHERE user_id = ? ORDER BY created_at DESC", (userId,))
        rows = c.fetchall()

    # 3. Convert rows to a list of dicts
    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "user_id": row["user_id"],
            "transcript": row["transcript"],
            "wpm_30": row["wpm_30"],
            "filler_count_30": row["filler_count_30"],
            "filler_pct_30": row["filler_pct_30"],
            "readability_60": row["readability_60"],
            "freq_60": row["freq_60"],
            "total_pauses": row["total_pauses"],
            "global_word_count": row["global_word_count"],
            "global_filler_count": row["global_filler_count"],
            "global_filler_pct": row["global_filler_pct"],
            "global_readability": row["global_readability"],
            "created_at": row["created_at"]
        })

    return JSONResponse(content=results)


@app.get("/history/{userId}/{rowId}", response_class=JSONResponse)
async def get_user_history_by_row(userId: str, rowId: int, request: Request):
    """
    Return a specific transcription entry in JSON format for the given userId and rowId.
    Expects 'Authorization: Bearer <userId>' to match the path userId.
    """
    # 1. Verify bearer token matches userId
    # auth_header = request.headers.get("Authorization")
    # if not auth_header or not auth_header.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    # token_user_id = auth_header[len("Bearer "):].strip()
    # if token_user_id != userId:
    #     raise HTTPException(status_code=403, detail="Token userId does not match path userId.")

    # 2. Query the database for a specific row belonging to userId
    with sqlite3.connect("transcriptions.db") as conn:
        conn.row_factory = sqlite3.Row  # Allows us to get dict-like rows
        c = conn.cursor()
        c.execute("SELECT * FROM transcriptions WHERE user_id = ? AND id = ?", (userId, rowId))
        row = c.fetchone()

    # 3. If no result is found, return 404
    if not row:
        raise HTTPException(status_code=404, detail="Transcription not found")

    # 4. Convert row to a dict and return as JSON
    result = {
        "id": row["id"],
        "user_id": row["user_id"],
        "transcript": row["transcript"],
        "wpm_30": row["wpm_30"],
        "filler_count_30": row["filler_count_30"],
        "filler_pct_30": row["filler_pct_30"],
        "readability_60": row["readability_60"],
        "freq_60": row["freq_60"],
        "total_pauses": row["total_pauses"],
        "global_word_count": row["global_word_count"],
        "global_filler_count": row["global_filler_count"],
        "global_filler_pct": row["global_filler_pct"],
        "global_readability": row["global_readability"],
        "created_at": row["created_at"]
    }

    return JSONResponse(content=result)

# =====================================================
#                 WEBSOCKET ENDPOINTS
# =====================================================

# 1) Example WebSocket Demo
@app.websocket("/ws/{userId}")
async def websocket_endpoint(websocket: WebSocket, userId: str):
    # Validate that the bearer token matches the userId in the path
    # authorize_user_id(websocket, userId)
    await websocket.accept()
    while True:
        # Echo a simple message
        await websocket.send_text(f"Hello {userId}, from WebSocket!")
        # Wait for data (blocking)
        await websocket.receive_text()

# 2) Vibrate WebSocket
@app.websocket("/vibrate")
async def vibrate_endpoint(websocket: WebSocket):
    # Validate that the bearer token matches the userId in the path
    # authorize_user_id(websocket, userId)
    await websocket.accept()
    try:
        while True:
            # Send "stop" continuously for 5 seconds
            stop_end_time = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < stop_end_time:
                await websocket.send_text("stop")
                await asyncio.sleep(0.1)

            # Send "vibrate" continuously for 5 seconds if vibrate is sent to it

            vibrate_end_time = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < vibrate_end_time:
                await websocket.send_text("vibrate")
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user")

# 3) Microphone Audio Handling WebSocket
@app.websocket("/microphone/{userId}")
async def microphone_endpoint(websocket: WebSocket, userId: str):
    """
    WebSocket endpoint for receiving real-time WebM audio.
    We store final transcript & analysis to SQLite under the userId.
    """
    # Validate that the bearer token matches the userId in the path
    # authorize_user_id(websocket, userId)
    await websocket.accept()
    print(f"Client (user_id={userId}) connected to /microphone")

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunks.append(data)
            print("Received audio chunk:", len(data))
    except WebSocketDisconnect:
        print(f"Client (user_id={userId}) disconnected.")
    except Exception as e:
        print(f"WebSocket error for user {userId}: {e}")
    finally:
        # Once disconnected, save the audio to a file and process
        with open(OUTPUT_WEBM_FILE, "wb") as webm_file:
            for chunk in audio_chunks:
                webm_file.write(chunk)
        print("Converting full audio to WAV...")
        convert_webm_to_wav(OUTPUT_WEBM_FILE, OUTPUT_WAV_FILE)
        # Process audio & do analysis
        await process_audio(OUTPUT_WAV_FILE, userId)
        # Clear the buffer for next time
        audio_chunks.clear()

# -------------------------------
# Convert Audio Helper
# -------------------------------
def convert_webm_to_wav(input_file, output_file):
    """Convert WebM (Opus) file to WAV."""
    try:
        audio = AudioSegment.from_file(input_file, format="webm")
        audio = audio.set_frame_rate(44100).set_channels(1)
        audio.export(output_file, format="wav")
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Audio conversion error: {e}")

# -------------------------------
# Process Audio (Transcription)
# -------------------------------
async def process_audio(filename: str, user_id: str):
    """
    Transcribe the WAV file with Whisper,
    then run metric calculations on the output,
    and store the result in the DB.
    """
    print(f"Transcribing audio for user={user_id} with Whisper pipeline ...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    whisper_model_name = "openai/whisper-tiny"

    def load_audio(audio_path: str):
        speech, sr_ = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sr_, 16000)
        speech = resampler(speech)
        return speech.squeeze()

    def get_long_transcription_whisper(audio_path: str, asr_pipe,
                                       return_timestamps=True, 
                                       chunk_length_s=10,
                                       stride_length_s=1):
        """Uses Hugging Face's pipeline for streaming large audio files."""
        return asr_pipe(
            load_audio(audio_path).numpy(), 
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s, 
            stride_length_s=stride_length_s
        )

    # Initialize HF pipeline for Whisper
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper_model_name,
        device=device
    )

    # Perform transcription
    output = get_long_transcription_whisper(
        filename, 
        asr_pipeline, 
        chunk_length_s=10, 
        stride_length_s=2
    )

    # Ensure "chunks" exists
    if "chunks" not in output or not isinstance(output["chunks"], list):
        print("No transcription chunks found.")
        return

    print("Transcription chunks found, analyzing ...")
    analyze_timestamped_transcript(output['chunks'], user_id)

# -------------------------------
# Metrics / Analysis
# -------------------------------
def rolling_wpm(history: List[RecognizedWord], current_time: float, window_size: float = 30.0) -> float:
    """Calculates rolling words per minute for the last `window_size` seconds."""
    window_words = [
        w for w in history 
        if current_time - window_size <= w.timestamp <= current_time
    ]
    if window_size <= 0:
        return 0.0
    wpm = (len(window_words) / window_size) * 60
    return round(wpm, 2)

def rolling_filler_stats(history: List[RecognizedWord], current_time: float, window_size: float = 30.0):
    """
    Returns (filler_count, filler_percentage) for the last `window_size` seconds.
    """
    window_words = [
        w.word.lower() for w in history
        if current_time - window_size <= w.timestamp <= current_time
    ]
    total = len(window_words)
    if total == 0:
        return 0, 0.0
    filler_count = sum(1 for word in window_words if word in FILLER_WORDS)
    filler_percentage = (filler_count / total) * 100
    return filler_count, round(filler_percentage, 2)

def detect_pauses(history: List[RecognizedWord], gap_threshold: float = 1.5) -> int:
    """
    Counts total number of pauses >= gap_threshold seconds in the entire history.
    """
    pause_count = 0
    for i in range(1, len(history)):
        gap = history[i].timestamp - history[i-1].timestamp
        if gap >= gap_threshold:
            pause_count += 1
    return pause_count

def rolling_readability(history: List[RecognizedWord], current_time: float, window_size: float = 60.0):
    """
    Flesch-Kincaid Grade Level for words in the last `window_size` seconds.
    Returns None if fewer than 10 words in that window.
    """
    window_words = [
        w.word for w in history
        if current_time - window_size <= w.timestamp <= current_time
    ]
    if len(window_words) < 10:
        return None
    text_chunk = " ".join(window_words)
    score = textstat.flesch_kincaid_grade(text_chunk)
    return round(score, 2)

def rolling_word_frequency(history: List[RecognizedWord], current_time: float, 
                           window_size: float = 60.0, top_n: int = 5):
    """
    Returns the most common `top_n` words in the last `window_size` seconds.
    """
    window_words = [
        w.word.lower() for w in history
        if current_time - window_size <= w.timestamp <= current_time
    ]
    freq = Counter(window_words)
    return freq.most_common(top_n)

# -------------------------------
# Transcript Analysis + DB Insert
# -------------------------------
def analyze_timestamped_transcript(chunks, user_id: str):
    """
    Parses the transcription chunks, calculates metrics,
    and stores the result in SQLite for the given user_id.
    """
    global speech_history
    speech_history.clear()

    # Step 1: Extract data from `chunks`
    for chunk in chunks:
        if not isinstance(chunk, dict) or "timestamp" not in chunk or "text" not in chunk:
            continue  # Skip invalid entries
        
        start_end = chunk["timestamp"]  # (start_time, end_time)
        text = chunk["text"].strip()
        if (not isinstance(start_end, tuple) or len(start_end) != 2 
            or not text):
            continue

        start, end = start_end
        if start is None or end is None or start >= end:
            continue

        # Tokenize text into words
        tokens = word_tokenize(text)
        if not tokens:
            continue

        # Distribute timestamps among tokens
        duration = max(0.001, end - start)
        gap_per_word = duration / len(tokens)

        for i, token in enumerate(tokens):
            word_time = start + (i * gap_per_word)
            speech_history.append(RecognizedWord(token, word_time))

    if not speech_history:
        print("No recognized words found. Cannot compute metrics.")
        return

    # Final Timestamp
    final_time = speech_history[-1].timestamp

    # Compute Rolling / Partial Metrics
    wpm_30 = rolling_wpm(speech_history, final_time, 30.0)
    filler_count_30, filler_pct_30 = rolling_filler_stats(speech_history, final_time, 30.0)
    readability_60 = rolling_readability(speech_history, final_time, 60.0)
    freq_60 = rolling_word_frequency(speech_history, final_time, 60.0, 5)
    total_pauses = detect_pauses(speech_history, gap_threshold=1.5)

    # Compute Global Metrics
    global_word_count = len(speech_history)
    global_text = " ".join(rw.word for rw in speech_history)
    if global_word_count >= 10:
        global_readability = round(textstat.flesch_kincaid_grade(global_text), 2)
    else:
        global_readability = None

    global_filler_count = sum(1 for rw in speech_history if rw.word.lower() in FILLER_WORDS)
    global_filler_pct = round((global_filler_count / global_word_count) * 100, 2) if global_word_count else 0.0

    # Debug info
    print("\n===== Real-Time Rolling Metrics (At End) =====")
    print(f"Final Timestamp: {final_time:.2f}s")
    print(f"Rolling WPM (30s): {wpm_30}")
    print(f"Filler (30s)   : count={filler_count_30}, pct={filler_pct_30}%")
    print(f"Readability(60s): {readability_60 if readability_60 is not None else 'N/A'}")
    print(f"Top Words(60s): {freq_60}")
    print(f"Total Pauses   : {total_pauses}")

    print("\n===== Global (Entire Speech) Metrics =====")
    print(f"Total Words    : {global_word_count}")
    print(f"Filler Count   : {global_filler_count} ({global_filler_pct}%)")
    print(f"Readability    : {global_readability if global_readability is not None else 'N/A'}")

    # Convert freq_60 to a storable string
    freq_60_str = str(freq_60)

    # Insert into SQLite
    with sqlite3.connect("transcriptions.db") as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO transcriptions (
                user_id,
                transcript,
                wpm_30,
                filler_count_30,
                filler_pct_30,
                readability_60,
                freq_60,
                total_pauses,
                global_word_count,
                global_filler_count,
                global_filler_pct,
                global_readability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            global_text,
            wpm_30,
            filler_count_30,
            filler_pct_30,
            readability_60 if readability_60 is not None else None,
            freq_60_str,
            total_pauses,
            global_word_count,
            global_filler_count,
            global_filler_pct,
            global_readability if global_readability is not None else None
        ))
        conn.commit()

    print(f"Successfully stored transcription & metrics in DB for user_id={user_id}.")

# -------------------------------
# Main Runner
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
