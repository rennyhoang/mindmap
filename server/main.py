import spacy
import uuid
import whisper
import numpy as np
import networkx as nx
from fastapi import FastAPI, WebSocket, HTTPException
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
            audio_chunk = await websocket.receive_bytes()
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            transcription = model.transcribe(audio_np, fp16=False)
            text = str(transcription["text"])
            transcript_store[session_id] += text + " "
            await websocket.send_text(text)

    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")

    finally:
        await websocket.send_text(f"Session ID: {session_id}")
        await websocket.close()


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