import uuid
import spacy
import whisper
import numpy as np
import networkx as nx
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_trf")
model = whisper.load_model("small.en")
transcript_store = {}


@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    transcript_store[session_id] = ""

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_chunk = message["bytes"]
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
                transcription = model.transcribe(audio_np, fp16=False)
                new_text = str(transcription["text"])
                transcript_store[session_id] += new_text + " "
                await websocket.send_text(transcript_store[session_id])
            elif "text" in message and message["text"] == "STOP":
                break
            else:
                session_id = message["text"]

    finally:
        await websocket.send_text(f"Session ID: {session_id}")
        await websocket.close()


@app.post("/uploadfile/{session_id}")
async def create_upload_file(file: UploadFile, session_id: str):
    if not session_id:
        session_id = str(uuid.uuid4())

    transcription = model.transcribe(file.filename)
    transcript_store[session_id] = transcription

    return {"session_id": session_id, "transcription": transcription}


@app.get("/graph/{session_id}")
async def generate_graph(session_id: str):
    if session_id not in transcript_store:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript = transcript_store[session_id]
    doc = nlp(transcript)
    G = nx.Graph()

    for entity in doc.ents:
        G.add_node(entity.text, label=entity.label_)

    for sentence in doc.sents:
        entities_in_sentence = [ent.text for ent in sentence.ents]
        for i in range(len(entities_in_sentence)):
            for j in range(i + 1, len(entities_in_sentence)):
                G.add_edge(entities_in_sentence[i], entities_in_sentence[j])

    pos = nx.spring_layout(G, k=200)

    nodes = [
        {
            "id": node,
            "position": {"x": float(x) * 200, "y": float(y) * 200},
            "data": {"label": node},
        }
        for node, (x, y) in pos.items()
    ]

    edges = [
        {
            "id": f"e-{source}-{target}",
            "source": str(source),
            "target": str(target),
        }
        for source, target in G.edges()
    ]

    return JSONResponse(content={"nodes": nodes, "edges": edges})
