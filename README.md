# LearnIt
LearnIt is a tool that transforms spoken or uploaded audio into structured mind maps. It combines automatic speech recognition (ASR), relation extraction, title generation via a fine-tuned BART model, and a Retrieval-Augmented Generation (RAG) chatbot for interactive Q&A over your transcript.
- A **FastAPI server** for speech-to-text, named entity recognition, relation extraction, title generation, vector storage, and a RAG chatbot.
- A **React client** using React Flow for interactive mind-map visualization and user interaction.

![image](https://github.com/user-attachments/assets/2fc99ae6-d7c2-4baa-8499-525ace005013)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Real-Time Recording & Streaming](#1-real-time-recording--streaming)
  - [2. Offline File Upload](#2-offline-file-upload)
  - [3. Interactive RAG Chatbot](#3-interactive-rag-chatbot)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Real-time ASR & TTS**  
  Record audio from microphone and transcribe on the fly.

- **Prerecorded Audio**  
  Process pre-recorded audio files (WAV/MP3/FLAC) for mind-map generation.

- **Relation Extraction**  
  Identify entities and their relations to define nodes and edges of the mind map.

- **Title Generation**  
  Summarize and title your session using a fine-tuned BART model.

- **RAG Chatbot**  
  Vectorize transcript segments (via FAISS) and answer user queries with retrieved context.

## Architecture

1. **Audio Ingestion**  
   - Live stream via RecordRTC and WebSockets
   - Audio File input

2. **Speech Recognition**  
   - OpenAI's Whisper + Voice Acitivity Detection

3. **Relation Extraction**  
   - SpaCy/Transformers pipeline → nodes & edges  

4. **Title Generation**  
   - Fine-tuned BART (Hugging Face) → session title  

5. **Vector Store & RAG**  
   - Store embeddings in Pinecone
   - LangChain for Q&A  

6. **Visualization**  
   - Interactive mindmap/graph using React Flow
