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
- [x] Preserve source and page metadata
- [ ] Document chunking
- [ ] Embedding generation and FAISS indexing
- [ ] RAG question-answering pipeline
- [ ] Source citation support
- [ ] Retrieval evaluation
- [ ] Complete automated testing
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

### Sample Document

Development uses the publicly available U.S. Department
of Energy sourcebook, "Improving Motor and Drive System
Performance."

Source:
https://www.energy.gov/sites/prod/files/2014/04/f15/amo_motors_sourcebook_web.pdf

Place a local copy at:

data/raw/motor_manual.pdf

PDF documents are excluded from Git tracking.
