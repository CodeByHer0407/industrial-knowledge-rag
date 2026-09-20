# Industrial Knowledge-Base RAG Assistant

A Retrieval-Augmented Generation (RAG) application designed to answer technical questions from industrial equipment documentation using Large Language Models (LLMs) and source-grounded retrieval.

## Project Status

🚧 Under active development.

## Problem Statement

Industrial equipment manuals and technical documents often contain large amounts of information, making it time-consuming to find specific troubleshooting procedures and maintenance instructions.

This project aims to provide a document-grounded question-answering system that retrieves relevant information and generates answers with supporting source references.

## Planned Features

- PDF document ingestion and text extraction
- Document chunking and metadata management
- Semantic search using embeddings and FAISS
- Hybrid retrieval using keyword and vector search
- LLM-based question answering
- Source citations with document names and page numbers
- REST API using FastAPI
- Retrieval evaluation and automated testing
- Docker deployment and CI/CD

## Technology Stack

Planned technologies:

- Python
- FastAPI
- LangChain
- FAISS
- Sentence Transformers
- PyMuPDF
- Docker
- Pytest

The technology stack will be finalized as development progresses.

## Getting Started

### Prerequisites

- Python 3.11
- Conda

### Create the environment

```bash
conda create -n industrial-rag python=3.11 -y
conda activate industrial-rag
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```
### Run the application

Start the FastAPI development server:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

Open the API documentation:

http://127.0.0.1:8765/docs

Health check endpoint:

http://127.0.0.1:8765/health

### Run tests

```powershell
python -m pytest -q
```

The current test suite verifies the health check endpoint.

## Development Roadmap

- [x] FastAPI backend setup
- [x] PDF ingestion and text extraction
- [x] Source and page metadata preservation
- [x] Document chunking with configurable overlap
- [ ] Embedding generation
- [ ] FAISS vector indexing
- [ ] Semantic search
- [ ] RAG question-answering pipeline
- [ ] Source citations
- [ ] Retrieval evaluation
- [ ] Docker deployment
- [ ] CI/CD integration

## PDF Ingestion

The application currently supports extracting text from
text-based PDF documents using PyMuPDF.

The extraction pipeline:
- Reads PDF documents page by page.
- Extracts text from non-empty pages.
- Preserves source filenames and page numbers.
- Validates file existence and extension.

## Document Chunking

The extracted PDF pages are split into smaller, overlapping
text chunks using a custom Python implementation.

### Configuration

- Chunk size: 120 words
- Chunk overlap: 25 words
- Splitting strategy: Page-aware, word-based splitting

### Features

- Configurable chunk size and overlap
- Preserves source filename and page number
- Generates a unique chunk ID per page and chunk position
- Skips empty pages
- Validates chunking configuration

### Initial Results

Using the motor systems sourcebook:

- Extracted pages: 93
- Generated chunks: 528

These are initial processing results. Retrieval accuracy will
be evaluated after implementing semantic search.

### Current Limitations

- Chunking is based on words rather than model tokens.
- Sentences may be split across chunk boundaries.
- Complex PDF tables and layouts are not preserved.
- Chunk IDs currently assume unique source filenames.

Future improvements will be guided by retrieval evaluation.


### Sample Document

Development uses the publicly available U.S. Department
of Energy sourcebook, "Improving Motor and Drive System
Performance."

Source:
https://www.energy.gov/sites/prod/files/2014/04/f15/amo_motors_sourcebook_web.pdf

Place a local copy at:

data/raw/motor_manual.pdf

PDF documents are excluded from Git tracking.
