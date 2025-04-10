import json
import os
import tempfile

import whisper
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("small.en")
transcript = ""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_bytes()
        if message:
            try:
                audio_bytes = message
                with tempfile.NamedTemporaryFile(
                    suffix=".ogg", delete=True
                ) as audio_chunk:
                    audio_chunk.write(audio_bytes)
                    audio_chunk.flush()
                    audio = whisper.pad_or_trim(
                        whisper.load_audio(
                            audio_chunk.name, sr=16000).reshape(1, -1)
                    )
                    transcription = whisper.transcribe(model, audio)
                    websocket.send(transcription)
            except Exception as e:
                print(e)
