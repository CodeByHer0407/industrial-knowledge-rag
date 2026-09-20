# Industrial Knowledge-Base RAG Assistant

A Retrieval-Augmented Generation (RAG) project for answering technical questions from industrial equipment documentation using semantic retrieval and Large Language Models (LLMs).

The project currently supports PDF ingestion, document chunking, embedding generation, and semantic search using FAISS. LLM-based answer generation is under development.

## Project Status

🚧 Under active development.

**Current milestone:** Semantic retrieval implemented and tested.

## Problem Statement

Industrial equipment manuals and technical documentation often contain large amounts of information, making it time-consuming for engineers to locate troubleshooting procedures, maintenance instructions, and technical specifications.

This project aims to develop a document-grounded question-answering system that retrieves relevant information and generates answers with supporting source references.

## Implemented Features

- PDF document ingestion and text extraction
- Page-aware document chunking with configurable overlap
- Source filename and page number preservation
- Embedding generation using Sentence Transformers
- Semantic search using FAISS
- Top-k retrieval with similarity scores and source metadata
- FastAPI backend foundation with health check endpoint
- Automated testing using Pytest

## Planned Features

- Persistent FAISS index storage
- Retrieval evaluation
- LLM-based question answering
- Source-grounded answer generation
- Hybrid retrieval using keyword and vector search
- REST API endpoints for document ingestion and querying
- Docker deployment
- CI/CD integration

## Technology Stack

### Current Technologies

- Python 3.11
- FastAPI
- PyMuPDF
- Sentence Transformers
- FAISS
- NumPy
- Pytest

### Planned Technologies

- LangChain
- Docker
- GitHub Actions

The technology stack may evolve as development progresses.

## Getting Started

### Prerequisites

- Python 3.11
- Conda

### 1. Create the Environment

```bash
conda create -n industrial-rag python=3.11 -y
conda activate industrial-rag
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the FastAPI Application

Start the development server:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

Open the API documentation:

http://127.0.0.1:8765/docs

Health check endpoint:

http://127.0.0.1:8765/health

Note: The FastAPI application currently exposes a health check endpoint. Document ingestion and question-answering endpoints will be added in subsequent milestones.

### 4. Run Automated Tests

```powershell
python -m pytest -q
```

Current test suite: 16 passing tests.

Tests cover:
- FastAPI health check
- PDF text extraction
- Empty and invalid document handling
- Document chunk size and overlap
- Source metadata preservation
- FAISS index construction
- Semantic search and input validation

## Document Ingestion

The application extracts text from text-based PDF documents using PyMuPDF.

The extraction pipeline:

1. Validates the input document.
2. Reads the PDF page by page.
3. Extracts text from non-empty pages.
4. Preserves source filenames and page numbers.

Image-only PDF documents are not currently supported.

### Sample Document

Development uses the publicly available U.S. Department of Energy sourcebook:

**Improving Motor and Drive System Performance**

Source:

https://www.energy.gov/sites/prod/files/2014/04/f15/amo_motors_sourcebook_web.pdf

Download the PDF and place it at:

```text
data/raw/motor_manual.pdf
```

The local PDF is excluded from Git tracking.

## Document Chunking

The extracted PDF pages are divided into smaller, overlapping text chunks using a custom Python implementation.

### Configuration

| Parameter | Value |
|-----------|-------|
| Chunk size | 120 words |
| Chunk overlap | 25 words |
| Splitting strategy | Page-aware, word-based |

### Features

- Configurable chunk size and overlap
- Source filename and page number preservation
- Chunk identifiers based on source, page, and position
- Empty-page handling
- Configuration validation

### Processing Results

Using the motor systems sourcebook:

| Metric | Result |
|--------|--------|
| Extracted pages | 93 |
| Generated chunks | 528 |

### Current Limitations

- Chunking uses word counts rather than model tokens.
- Sentences may be split across chunk boundaries.
- Complex PDF tables and layouts are not preserved.
- Chunk identifiers currently assume unique source filenames.
- PDF extraction may produce broken words or formatting artifacts.

Future improvements will be guided by retrieval evaluation.

## Semantic Retrieval

The application implements semantic search using Sentence Transformers and FAISS.

### Embedding Model

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimensions: 384
- Execution: Local CPU
- Embeddings: L2-normalized
- Output data type: float32

### Vector Search

- Index: FAISS IndexFlatIP
- Search: Exact nearest-neighbor search
- Similarity: Inner product of normalized vectors (cosine similarity)
- Retrieval: Configurable top-k
- Metadata: Source filename, page number, and chunk ID

### Processing Results

| Metric | Result |
|--------|--------|
| Indexed chunks | 528 |
| Embedding dimensions | 384 |
| Embedding matrix | 528 × 384 |
| Invalid embedding values | None detected |

### Example Query

```text
How can motor efficiency be improved?
```

The semantic search pipeline retrieves relevant document passages along with similarity scores, source filenames, and page numbers.

Retrieval quality has not yet been formally evaluated.

**Current limitation:** The FAISS index is maintained in memory. Persistent index storage will be implemented in the next milestone.

## Testing

The project currently includes 16 passing automated tests.

Run the complete test suite:

```powershell
python -m pytest -q
```

Tests are designed to validate individual components without requiring the industrial PDF or downloading the embedding model for every test.

## Development Roadmap

- [x] FastAPI backend setup
- [x] PDF ingestion and text extraction
- [x] Document chunking with metadata
- [x] Embedding generation
- [x] FAISS vector indexing
- [x] Semantic retrieval
- [x] Initial automated tests
- [ ] Persistent FAISS index storage
- [ ] Retrieval evaluation
- [ ] LLM-based question answering
- [ ] Hybrid retrieval
- [ ] Docker deployment
- [ ] CI/CD integration

## Project Notes

This project is being developed incrementally, with each component implemented, tested, and documented before introducing additional functionality.

Retrieval evaluation results and deployment instructions will be added as the project progresses.