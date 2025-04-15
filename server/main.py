import uuid
import tempfile
from itertools import combinations

import spacy
import whisper
import numpy as np
import networkx as nx
from fastapi import FastAPI, File, WebSocket, HTTPException, UploadFile
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
lemmatizer = nlp.get_pipe("lemmatizer")
model = whisper.load_model("small.en")
transcript_store = {}

EXCLUDED_ENTITY_LABELS = {
    "DATE",  # e.g., "January 3rd", "10/10/2020"
    "TIME",  # e.g., "3:00 PM"
    "PERCENT",  # e.g., "50%"
    "MONEY",  # e.g., "$100", "1 billion dollars"
    "QUANTITY",  # e.g., "100 kg", "3 liters"
    "ORDINAL",  # e.g., "first", "third"
    "CARDINAL",  # e.g., "one", "200"
}


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


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    print("WE IN HERE")
    session_id = str(uuid.uuid4())
    contents = await file.read()

    print("WE ACTUALLY IN HERE")
    with tempfile.NamedTemporaryFile(delete=True, mode="w+b") as tmp_file:
        tmp_file.write(contents)
        print("WE WROTE TO THE FILE")
        transcription = model.transcribe(tmp_file.name)
        print("WE ARE DONE TRANSCRIBING")
        transcript_store[session_id] = transcription["text"]

    return JSONResponse(
        status_code=200,
        content={"session_id": session_id, "transcription": transcription["text"]},
    )


@app.get("/graph/{session_id}")
async def generate_graph(session_id: str):
    if session_id not in transcript_store:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript = transcript_store[session_id]
    doc = nlp(transcript)
    graph = nx.Graph()

    for entity in doc.ents:
        if entity.label_ not in EXCLUDED_ENTITY_LABELS:
            graph.add_node(entity.lemma_.lower(), label=entity.label_)

    for sentence in doc.sents:
        entities_in_sentence = [
            ent.lemma_
            for ent in sentence.ents
            if ent.label_ not in EXCLUDED_ENTITY_LABELS
        ]
        for entity1, entity2 in combinations(entities_in_sentence, 2):
            graph.add_edge(entity1, entity2)

    pos = nx.spring_layout(graph, k=200)

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
        for source, target in graph.edges()
    ]

    return JSONResponse(status_code=200, content={"nodes": nodes, "edges": edges})
