# MindMap Helper

MindMap Helper is a tool that transforms spoken or uploaded audio into structured mind maps. It combines automatic speech recognition (ASR), relation extraction, title generation via a fine-tuned BART model, and a Retrieval-Augmented Generation (RAG) chatbot for interactive Q&A over your transcript.

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
  Record audio from microphone, transcribe on the fly, and optionally stream synthesized speech of the transcript.

- **Offline Upload**  
  Process pre-recorded audio files (WAV/MP3/FLAC) for batch mind-map generation.

- **Relation Extraction**  
  Identify entities and their relations to define nodes and edges of the mind map.

- **Title Generation**  
  Summarize and title your session using a fine-tuned BART model.

- **RAG Chatbot**  
  Vectorize transcript segments (via FAISS) and answer user queries with retrieved context.

## Architecture

1. **Audio Ingestion**  
   - Live stream via PyAudio/sounddevice  
   - File input via `ffmpeg`  

2. **Speech Recognition**  
   - Whisper or SpeechRecognition → raw transcript  

3. **Relation Extraction**  
   - SpaCy/Transformers pipeline → nodes & edges  

4. **Title Generation**  
   - Fine-tuned BART (Hugging Face) → session title  

5. **Vector Store & RAG**  
   - Encode transcript per chunk (sentence/paragraph)  
   - Store embeddings in FAISS  
   - LangChain/OpenAI for Q&A  

6. **Visualization**  
   - Export mind map (Graphviz/D3.js)  

## Prerequisites

- Python 3.8+
- `pip` or `poetry`
- `ffmpeg` (for audio conversions)
- Microphone (for live mode)

## Installation

```bash
git clone https://github.com/yourusername/mindmap-helper.git
cd mindmap-helper
python -m venv venv
source venv/bin/activate    # macOS/Linux
# .\venv\Scripts\activate   # Windows
pip install -r requirements.txt
