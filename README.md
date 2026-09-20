# Industrial Knowledge-Base RAG Assistant

A Retrieval-Augmented Generation (RAG) project for building a document-grounded question-answering system for industrial technical documentation.

The current implementation supports PDF ingestion, page-aware document chunking, embedding generation, FAISS semantic search, and persistent vector storage.

LLM-based answer generation, retrieval evaluation, and deployment are under development.

---

## 1. Project Status

🚧 **Under Active Development**

**Current milestone:** Semantic retrieval and persistent FAISS index storage implemented and tested.

**Next milestone:** Retrieval evaluation.

The project is being developed incrementally, with each component implemented, tested, and documented before introducing additional functionality.

---

## 2. Problem Statement

Industrial equipment manuals and technical documentation often contain large amounts of information, making it time-consuming for engineers to locate troubleshooting procedures, maintenance instructions, and technical specifications.

Traditional keyword-based searches may miss relevant information when user queries and technical documents use different terminology.

This project aims to develop a document-grounded question-answering system that:

- Retrieves relevant information from industrial documentation.
- Supports natural-language queries.
- Preserves source information for traceability.
- Generates answers grounded in retrieved documentation.

The current implementation focuses on document processing and semantic retrieval. Answer generation will be introduced in a later milestone.

---

## 3. Implemented Features

### Document Processing

- PDF document ingestion and text extraction using PyMuPDF.
- Page-aware document chunking with configurable chunk size and overlap.
- Preservation of source filenames and page numbers.
- Generation of chunk identifiers.
- Validation of input files and chunking configurations.

### Semantic Retrieval

- Text embedding generation using Sentence Transformers.
- Local CPU-based embedding inference.
- FAISS vector indexing.
- Cosine similarity search using normalized embeddings.
- Configurable top-k retrieval.
- Retrieval results containing similarity scores and source metadata.

### Vector Index Persistence

- Persistent FAISS index storage.
- JSON-based metadata storage.
- Embedding model compatibility validation during loading.
- Reusable scripts for index generation and semantic search.
- Search without regenerating document embeddings.

### Backend and Testing

- FastAPI backend foundation.
- Health check endpoint.
- Automated tests using Pytest.
- 18 passing tests covering the implemented components.

---

## 4. Planned Features

- Retrieval evaluation using a manually labeled question dataset.
- LLM-based question answering.
- Source-grounded answer generation.
- Hybrid retrieval using keyword and vector search.
- REST API endpoints for document ingestion and querying.
- Improved document preprocessing and chunking.
- Docker containerization.
- CI/CD integration using GitHub Actions.

---

## 5. Technology Stack

### Current Technologies

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.11 |
| Backend | FastAPI |
| PDF Processing | PyMuPDF |
| Embedding Framework | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Search | FAISS |
| Numerical Computing | NumPy |
| Testing | Pytest |
| Version Control | Git and GitHub |

### Planned Technologies

- LangChain
- Docker
- GitHub Actions

The technology stack may evolve as additional features are implemented.

---

## 6. System Architecture

The current implementation separates document indexing from semantic search.

### 6.1 Document Indexing Pipeline

```text
Industrial PDF Documentation
           |
           v
    PDF Text Extraction
        (PyMuPDF)
           |
           v
   Page-Aware Chunking
   (120 words, 25 overlap)
           |
           v
    Text Embedding Model
      (MiniLM-L6-v2)
           |
           v
       FAISS Index
      (IndexFlatIP)
           |
           v
      Save to Disk
           |
     +-----+-----+
     |           |
     v           v
 index.faiss  metadata.json
```

The indexing process extracts document text, generates chunks and embeddings, and saves the FAISS index and metadata.

### 6.2 Semantic Search Pipeline

```text
       User Question
             |
             v
      Embedding Model
       (MiniLM-L6-v2)
             |
             v
      Query Embedding
             |
             v
    Load Saved FAISS Index
             |
             v
      Similarity Search
             |
             v
     Top-K Matching Chunks
             |
             v
    Retrieved Text + Score
    + Source + Page Number
```

The search pipeline reuses the saved document index and generates an embedding only for the incoming question.

LLM-based answer generation is not yet implemented.

---

## 7. Getting Started

### 7.1 Prerequisites

- Python 3.11
- Conda
- Git
- Internet access for the initial embedding model download

### 7.2 Clone the Repository

```powershell
git clone https://github.com/CodeByHer0407/industrial-knowledge-rag.git

cd industrial-knowledge-rag
```

### 7.3 Create the Conda Environment

```powershell
conda create -n industrial-rag python=3.11 -y

conda activate industrial-rag
```

Verify the Python version:

```powershell
python --version
```

### 7.4 Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The requirements file includes the dependencies needed for the current implementation.

### 7.5 Download the Sample Document

Development uses the publicly available U.S. Department of Energy sourcebook:

**Improving Motor and Drive System Performance**

Source:

https://www.energy.gov/sites/prod/files/2014/04/f15/amo_motors_sourcebook_web.pdf

Create the following directory if it does not exist:

```powershell
New-Item -ItemType Directory -Force data/raw
```

Download the PDF and save it as:

```text
data/raw/motor_manual.pdf
```

The PDF is excluded from Git tracking.

### 7.6 Build the FAISS Index

Run:

```powershell
python -m scripts.build_index
```

This script performs the following operations:

1. Extracts text from the PDF.
2. Generates overlapping document chunks.
3. Loads the embedding model.
4. Generates document embeddings.
5. Builds the FAISS index.
6. Saves the index and metadata to disk.

The first execution may download the embedding model from Hugging Face.

The generated files are stored in:

```text
data/index/
├── index.faiss
└── metadata.json
```

The index is rebuilt when the indexing script is executed.

### 7.7 Search the Documentation

Once the index has been generated, run:

```powershell
python -m scripts.search_index "How can motor efficiency be improved?"
```

The script:

1. Loads the saved FAISS index.
2. Loads the embedding model.
3. Converts the question into an embedding.
4. Retrieves the five most similar document chunks.
5. Displays the retrieved passages and metadata.

Each result contains:

- Similarity score
- Source filename
- PDF page number
- Chunk ID
- Retrieved text

Document embeddings are not regenerated during search.

### 7.8 Run the FastAPI Application

Start the development server:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --reload
```

Open the interactive API documentation:

http://127.0.0.1:8765/docs

Health check endpoint:

http://127.0.0.1:8765/health

**Note:** The FastAPI application currently exposes a health check endpoint. Semantic search is available through the CLI script. Document ingestion and question-answering API endpoints will be implemented later.

### 7.9 Run Automated Tests

```powershell
python -m pytest -q
```

Latest reported local test result:

```text
18 passed
```

---

## 8. Document Ingestion

The document ingestion component is implemented in:

```text
app/ingestion.py
```

It extracts text from text-based PDFs using PyMuPDF.

### Extraction Workflow

1. Validate the input file.
2. Open the PDF document.
3. Extract text page by page.
4. Skip pages without extractable text.
5. Preserve source filename and page number.

### Example Output

```python
{
    "source": "motor_manual.pdf",
    "page": 1,
    "text": "Cover Page - Motor A SOURCEBOOK..."
}
```

Page numbers refer to PDF page positions, which may differ from printed page numbers in the document.

### Current Limitations

- Image-only scanned PDFs are not supported.
- Complex layouts and tables may not be extracted correctly.
- PDF extraction may introduce whitespace and word-formatting artifacts.

---

## 9. Document Chunking

The document chunking component is implemented in:

```text
app/chunking.py
```

It divides extracted PDF pages into smaller, overlapping text chunks.

### Chunking Configuration

| Parameter | Value |
|-----------|-------|
| Chunk size | 120 words |
| Chunk overlap | 25 words |
| Step size | 95 words |
| Strategy | Page-aware, word-based splitting |

Chunk overlap helps preserve context near chunk boundaries.

Each page is processed independently so that chunks retain a single source page number.

### Features

- Configurable chunk size.
- Configurable overlap.
- Source and page metadata preservation.
- Chunk ID generation.
- Empty-page handling.
- Configuration validation.

### Example Chunk

```python
{
    "chunk_id": "motor_manual.pdf_p29_c3",
    "source": "motor_manual.pdf",
    "page": 29,
    "text": "are specified, configured, and maintained properly..."
}
```

### Processing Results

Using the motor systems sourcebook:

| Metric | Result |
|--------|--------|
| Extracted pages | 93 |
| Generated chunks | 528 |

### Current Limitations

- Chunking uses word counts rather than model tokens.
- Sentences may be divided across chunk boundaries.
- Complex tables and document layouts are not preserved.
- Chunk identifiers currently assume unique source filenames.
- Overlapping chunks may introduce duplicate information during retrieval.

Future improvements will be guided by retrieval evaluation.

---

## 10. Embedding Generation

The embedding component is implemented in:

```text
app/embeddings.py
```

### Embedding Model

The application uses:

`sentence-transformers/all-MiniLM-L6-v2`

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dimensions | 384 |
| Execution device | CPU |
| Output type | float32 |
| Normalization | L2 normalization |
| Batch size | 16 |

The embedding model converts document chunks and user questions into numerical vectors.

The same embedding model is used during indexing and querying.

### Embedding Results

| Metric | Result |
|--------|--------|
| Input chunks | 528 |
| Embedding dimensions | 384 |
| Embedding matrix | 528 × 384 |
| Invalid embedding values | None detected |

The generated embeddings were checked for NaN and infinite values before indexing.

The model may truncate inputs exceeding its supported input length. Token-length validation is a planned preprocessing improvement.

---

## 11. Semantic Retrieval

The vector search component is implemented in:

```text
app/vector_store.py
```

### FAISS Configuration

| Parameter | Value |
|-----------|-------|
| Index | IndexFlatIP |
| Search type | Exact nearest-neighbor |
| Similarity | Inner product |
| Vector normalization | L2 |
| Indexed vectors | 528 |
| Dimensions | 384 |

Because both document and query vectors are normalized, inner-product similarity corresponds to cosine similarity, subject to floating-point precision.

### Retrieval Workflow

1. Receive a natural-language question.
2. Generate its embedding using the same model.
3. Normalize the query vector.
4. Search the FAISS index.
5. Retrieve the top-k matching vector positions.
6. Map vector positions to document chunks.
7. Return text and source metadata.

### Example Query

```text
How can motor efficiency be improved?
```

### Observed Retrieval Results

The initial semantic search returned passages from the following PDF pages:

| Rank | PDF Page | Similarity |
|------|----------|------------|
| 1 | 29 | 0.6980 |
| 2 | 43 | 0.6900 |
| 3 | 11 | 0.6861 |
| 4 | 71 | 0.6704 |
| 5 | 29 | 0.6528 |

The retrieved passages include information related to motor configuration, operating load, efficiency, and energy consumption.

Some retrieved passages are general background information or bibliography entries rather than direct answers.

These observations will inform future retrieval improvements.

**Important:** Similarity scores represent vector similarity, not retrieval accuracy or answer correctness.

Retrieval quality has not yet been formally evaluated.

---

## 12. FAISS Index Persistence

The application supports saving and loading the FAISS index and its associated metadata.

This separates document indexing from user queries.

### 12.1 Index Building

Run:

```powershell
python -m scripts.build_index
```

The script generates the document embeddings, builds the FAISS index, and saves the resulting artifacts.

### 12.2 Saved Artifacts

| File | Purpose |
|------|---------|
| `index.faiss` | Stores the FAISS vector index |
| `metadata.json` | Stores chunk text, source information, and model metadata |

Both files are stored in:

```text
data/index/
```

Generated index files are excluded from Git tracking.

### 12.3 Index Loading

Run:

```powershell
python -m scripts.search_index "How can motor efficiency be improved?"
```

The script loads the saved index and generates an embedding only for the incoming question.

It does not repeat PDF extraction, document chunking, or document embedding generation.

### 12.4 Validation

The implementation validates:

- Index and metadata file existence.
- Embedding model compatibility.
- Embedding dimensions.
- Vector and chunk counts.

### 12.5 Persistence Verification

Persistence was tested by saving the complete index and loading it in a separate Python process.

| Metric | Result |
|--------|--------|
| Loaded vectors | 528 |
| Vector dimensions | 384 |
| Retrieved results | 5 |
| Metadata preservation | Verified |

The reloaded index returned the same top-five passages and similarity scores observed before persistence.

### 12.6 Security Considerations

Only locally generated, trusted FAISS index files should be loaded.

Metadata validation checks consistency but does not authenticate an index file or make untrusted FAISS files safe to load.

---

## 13. Testing

The project currently includes 18 passing automated tests.

Run:

```powershell
python -m pytest -q
```

### Test Coverage

| Component | Tests |
|-----------|-------|
| FastAPI | Health check |
| PDF Ingestion | Text extraction, empty pages, missing files, invalid files |
| Document Chunking | Chunk size, overlap, metadata, empty pages, invalid configuration |
| Vector Store | Index construction, search, metadata mapping, input validation |
| Persistence | Save/load verification and model compatibility |

Tests use temporary PDF documents and synthetic vectors where appropriate.

This avoids requiring the full industrial PDF or downloading the embedding model during every test run.

### Latest Reported Test Result

```text
18 passed
```

---

## 14. Development Roadmap

### Completed

- [x] Repository initialization and environment setup
- [x] FastAPI backend foundation
- [x] PDF ingestion and text extraction
- [x] Source and page metadata preservation
- [x] Document chunking
- [x] Embedding generation
- [x] FAISS vector indexing
- [x] Semantic retrieval
- [x] Initial automated tests
- [x] Persistent FAISS index storage
- [x] Reusable indexing and search scripts

### In Progress / Upcoming

- [ ] Create a retrieval evaluation dataset
- [ ] Implement Recall@K, Hit Rate@K, and MRR
- [ ] Analyze retrieval failures
- [ ] Improve document preprocessing
- [ ] Implement LLM-based question answering
- [ ] Add source-grounded answer generation
- [ ] Implement hybrid retrieval
- [ ] Expose retrieval and question-answering APIs
- [ ] Containerize the application using Docker
- [ ] Implement CI/CD using GitHub Actions
- [ ] Finalize project documentation and demonstration

---

## 15. Known Limitations

The current implementation has the following limitations:

- Supports text-based PDFs only.
- Processes one configured sample document through the current indexing script.
- Uses word-based chunking rather than token-aware splitting.
- May lose formatting information during PDF extraction.
- Does not yet provide formal retrieval evaluation metrics.
- Does not yet generate answers using an LLM.
- Semantic search is currently accessible through a CLI script rather than an API endpoint.
- Requires rebuilding the index when the underlying documents or indexing configuration change.

These limitations will be addressed incrementally based on implementation priorities and evaluation results.

---

## 16. Data Source

The project uses the U.S. Department of Energy sourcebook:

**Improving Motor and Drive System Performance**

Official document:

https://www.energy.gov/sites/prod/files/2014/04/f15/amo_motors_sourcebook_web.pdf

The document is used as a development reference.

The repository does not redistribute the source PDF. Users should obtain the document from its original source and observe any applicable usage conditions.

---

## 17. Project Notes

This project is being developed incrementally to demonstrate practical AI/ML engineering concepts, including:

- Document ingestion and preprocessing
- Text embeddings
- Vector similarity search
- Metadata management
- Persistent vector storage
- Automated testing
- Retrieval evaluation
- LLM application development

Implementation decisions, evaluation results, and deployment instructions will be updated as development progresses.