import getpass
import os
import uuid
import tempfile
import itertools
from itertools import combinations

import spacy
import whisper
import numpy as np
import networkx as nx
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from fastapi import FastAPI, File, WebSocket, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pinecone import ServerlessSpec, Pinecone
from scipy.spatial.distance import cosine
import tiktoken

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_md")
lemmatizer = nlp.get_pipe("lemmatizer")
model = whisper.load_model("small.en")
transcript_store = {}

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
model_name = 'multilingual-e5-large'
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
print("Indexes" + str(pc.list_indexes()))

EXCLUDED_ENTITY_LABELS = {
    "DATE",  # e.g., "January 3rd", "10/10/2020"
    "TIME",  # e.g., "3:00 PM"
    "PERCENT",  # e.g., "50%"
    "MONEY",  # e.g., "$100", "1 billion dollars"
    "QUANTITY",  # e.g., "100 kg", "3 liters"
    "ORDINAL",  # e.g., "first", "third"
    "CARDINAL",  # e.g., "one", "200"
}


class GraphRequest(BaseModel):
    sessionId: str
    transcript: str

class Question(BaseModel):
    sessionId: str
    question: str

def upload_text(session_id, text):
    index_name = "learnit" 
    if index_name not in pc.list_indexes():
        pc.create_index(name = index_name, dimension=1536, vector_type="dense", metric="cosine", spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),)
    index = pc.Index(index_name)
    for c in chunk_text(text):
        upsert_text(index, "chunk_" + str(uuid.uuid4()), c, session_id)

def chunk_text(text: str,
                        max_words: int = 200,
                        overlap: int = 50) -> list[str]:
    """
    Split text into chunks of up to max_words words,
    with `overlap` words overlapping between chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # advance but keep `overlap` words for context
        start += max_words - overlap

    return chunks

def embed_text(text: str) -> list[float]:
    """Generate an OpenAI embedding for the given text."""
    resp = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return resp.data[0].embedding

def upsert_text(index, id: str, text: str, namespace):
    """Embed text and upsert into Pinecone with metadata."""
    vector = embed_text(text)
    index.upsert(
        vectors=[
            (id, vector, {"text": text})
        ],
        namespace=namespace
    )

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
        upload_text(session_id, transcript_store[session_id])


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=True, mode="w+b") as tmp_file:
        tmp_file.write(contents)
        transcription = model.transcribe(tmp_file.name)
        transcript_store[session_id] = transcription["text"]


    upload_text(session_id, transcript_store[session_id])

    return JSONResponse(
        status_code=200,
        content={"session_id": session_id, "transcription": transcription["text"]},
    )

@app.post("/qa/")
async def answer_question(question: Question):
    sessionId = question.sessionId
    question_text = question.question
    question_vector = embed_text(question_text)
    index = pc.Index("learnit")
    res = index.query(
        namespace = sessionId,
        vector = question_vector,
        top_k = 2,
        include_metadata=True,
        include_values=False
    )
    context = "\n\n---\n\n".join([m["metadata"]["text"] for m in res["matches"]])
    print(context)
    messages = [
        SystemMessage("You are a helpful assistant. Use the provided context to answer."),
        HumanMessage(f"Context:\n{context}\n\n" + f"###\n\n" + f"Question: {question_text}")
    ]
    answer = llm.invoke(messages).content

    return JSONResponse(
        status_code=200,
        content={
            "answer": answer
        },
    )


@app.post("/graph/")
async def generate_graph(graph_request: GraphRequest):
    session_id = graph_request.sessionId
    transcript = graph_request.transcript

    if not session_id or session_id not in transcript_store:
        session_id = str(uuid.uuid4())
        transcript_store[session_id] = transcript

    messages = [
        SystemMessage("Return a short title based on the following text"),
        HumanMessage(transcript),
    ]
    topic = llm.invoke(messages).content
    print(topic)

    doc = nlp(transcript)
    graph = nx.Graph()
    entities = []
    similarity_threshold = 0.8

    for entity in doc.ents:
        if entity.label_ in EXCLUDED_ENTITY_LABELS or entity in entities:
            continue
        too_similar = False
        for other in entities:
            print(entity.similarity(other), entity.lemma_, other.lemma_)
            if entity.similarity(other) > similarity_threshold:
                too_similar = True
                break
        if not too_similar:
            entities.append(entity)
    
    for entity in entities:
        graph.add_node(entity.lemma_.lower(), label=entity.label_)

    '''
    for sentence in doc.sents:
        entities_in_sentence = [
            ent.lemma_.lower()
            for ent in sentence.ents
            if ent in entities
        ]
        for entity1, entity2 in combinations(entities_in_sentence, 2):
            graph.add_edge(entity1, entity2)
    
    '''
    sentence_entities = []
    for sentence in doc.sents:
        entities_in_current_sentence = [
            ent.lemma_.lower()
            for ent in sentence.ents
            if ent in entities
        ]
        sentence_entities.append(entities_in_current_sentence)

    window_size = 8 # The window includes the current sentence and 3 previous ones
    for i in range(len(sentence_entities)):
        start_index = max(0, i - window_size + 1)

        current_window_entities = []
        for j in range(start_index, i + 1):
            current_window_entities.extend(sentence_entities[j])

        unique_window_entities = list(set(current_window_entities))
        for entity1, entity2 in itertools.combinations(unique_window_entities, 2):
            if entity1 != entity2:
                graph.add_edge(entity1, entity2)



    pos = nx.spring_layout(graph)

    nodes = [
        {
            "id": node,
            "position": {"x": float(x) * 1500, "y": float(y) * 1500},
            "data": {"label": node},
        }
        for node, (x, y) in pos.items()
    ]

    edges = [
        {
            "id": f"e-{source}-{target}",
            "source": str(source),
            "target": str(target),
            "label": llm.invoke(
                [
                    SystemMessage(
                        "Return a brief phrase about how the following two topics are related. Limit it to three words max."
                    ),
                    HumanMessage(str(source) + " " + str(target)),
                ]
            ).content,
        }
        for source, target in graph.edges()
    ]

    return JSONResponse(
        status_code=200,
        content={
            "session_id": session_id,
            "nodes": nodes,
            "edges": edges,
            "title": topic,
        },
    )