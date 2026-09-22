# Industrial Knowledge-Base RAG Assistant

A local-first Retrieval-Augmented Generation (RAG) application for asking natural-language questions about industrial technical documentation. It extracts text from a PDF, retrieves relevant passages with Sentence Transformers and FAISS, generates answers using a local Ollama model, and reports source citations and citation-reference validation through a CLI and FastAPI.

**Current scope:** A working, single-document prototype using a publicly available industrial motor sourcebook. This is a portfolio project, not a production-validated technical decision system.

## Features

- **Document ingestion:** Extract text from text-based PDFs using PyMuPDF; retain source filename and PDF page number.
- **Page-aware chunking:** Split each page into 120-word chunks with 25-word overlap and retain chunk IDs.
- **Targeted preprocessing:** Remove identified standalone page-header chunks before indexing.
- **Semantic retrieval:** Generate normalized `all-MiniLM-L6-v2` embeddings and search a persistent FAISS `IndexFlatIP` index.
- **Document-grounded generation:** Build an evidence-only prompt and use `llama3.2:3b` through local Ollama. A separate OpenAI client is implemented but **not required** for local use.
- **Traceability:** Return retrieved passages, scores, source filenames, page numbers, and bracketed source citations.
- **Citation-reference checks:** Detect citations to source/page pairs that were not retrieved and flag missing or malformed bracketed citations. Explicit insufficient-evidence responses are marked as abstentions.
- **Interfaces and tests:** CLI, FastAPI `/health` and `/ask` endpoints, retrieval evaluation, and automated pytest tests.

## Architecture

```mermaid
flowchart TD
    PDF[PDF sourcebook] --> Extract[PyMuPDF text extraction]
    Extract --> Chunk[Page-aware chunking]
    Chunk --> Filter[Header-only chunk filtering]
    Filter --> Embed[MiniLM embeddings]
    Embed --> Index[FAISS index + JSON metadata]
    Question[User question] --> QueryEmbed[Question embedding]
    QueryEmbed --> Index
    Index --> Retrieve[Top-k passages + source metadata]
    Retrieve --> Prompt[Evidence-grounded prompt]
    Question --> Prompt
    Prompt --> LLM[Local Ollama: Llama 3.2 3B]
    LLM --> Validate[Citation-reference validation / abstention status]
    Validate --> Output[Answer + sources + status + validation report]
```

Index construction is separate from querying. The FastAPI application lazily loads and caches the saved FAISS index and embedding model when `/ask` is first called. The `/health` endpoint checks API liveness, not index or Ollama readiness.

## Tech stack

| Component | Technology |
| --- | --- |
| Language / API | Python 3.11, FastAPI, Pydantic |
| PDF extraction | PyMuPDF |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` (384 dimensions) |
| Vector search | FAISS `IndexFlatIP` with L2-normalized vectors (cosine similarity) |
| Local answer generation | Ollama, `llama3.2:3b` |
| Optional cloud client | OpenAI (not used in the local workflow) |
| Testing | pytest, FastAPI TestClient / HTTPX |

## Quick start: Windows PowerShell

**Prerequisites:** Git, Conda with Python 3.11, and [Ollama for Windows](https://ollama.com/download/windows). An internet connection is needed for the initial Python dependency, embedding-model, and Ollama-model downloads. Local inference does not require OpenAI credits.

### 1. Clone and install

```powershell
git clone https://github.com/CodeByHer0407/industrial-knowledge-rag.git
cd industrial-knowledge-rag
conda create -n industrial-rag python=3.11 -y
conda activate industrial-rag
python -m pip install -r requirements.txt
```

### 2. Obtain the sample PDF

This project uses the U.S. Department of Energy sourcebook **Improving Motor and Drive System Performance**:

https://www.energy.gov/sites/prod/files/2014/04/f15/amo_motors_sourcebook_web.pdf

Create the input directory, download the PDF from the source above, and save it with this exact name:

```powershell
New-Item -ItemType Directory -Force data/raw
# Save the downloaded PDF as: data/raw/motor_manual.pdf
```

The PDF and generated index are deliberately excluded from Git; they are not bundled with the repository. Check the source's terms before redistributing its content.

### 3. Build the FAISS index

```powershell
python -m scripts.build_index
```

This extracts pages, makes overlapping chunks, filters identified header-only chunks, embeds the remaining text, and saves `data/index/index.faiss` and `data/index/metadata.json`. In the documented sample run, 528 generated chunks were reduced to **524 indexed chunks** after removing four header-only chunks. If you change the document or indexing configuration, rebuild the index.

### 4. Download the local generation model

```powershell
ollama run llama3.2:3b
```

The first run downloads the model; enter `/bye` to exit its chat. Keep the Ollama service running when you use the application. On Windows, the Ollama app normally manages the local server; if the app is not running, start it before making requests.

### 5. Ask a question locally

```powershell
# Retrieval + prompt preview; no LLM call:
python -m scripts.ask "How can motor efficiency be improved?"

# Full local RAG answer (no paid API):
python -m scripts.ask "What happens when a motor operates below 40% of full load?" --provider ollama --live

# See the exact retrieved prompt:
python -m scripts.preview_rag_prompt "How can motor efficiency be improved?"
```

The CLI displays retrieved source pages and similarity scores. In live mode, it also prints the generated answer and citation-reference validation. A valid citation reference **does not prove** that the cited text supports every claim.

### 6. Run the API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API documentation. For example, send this JSON body to `POST /ask`:

```json
{
  "question": "What happens when a motor operates below 40% of full load?",
  "top_k": 5
}
```

The response contains `question`, `answer`, `sources`, `answer_status` (`answered`, `abstained`, or `no_context`), and `citation_validation` (a report or `null` when not applicable). `GET /health` checks whether the API is responding. The current `/ask` implementation uses local Ollama; it does not call OpenAI.

## Evaluation

The repository includes `eval/questions.json` and `scripts/evaluate_retrieval.py`, with a small development set of **10 technical questions** and manually identified supporting chunk IDs.

```powershell
python -m scripts.evaluate_retrieval
```

| Retrieval metric | Original index (528 chunks) | Filtered index (524 chunks) |
| --- | ---: | ---: |
| Hit Rate@5 | 1.000 | 1.000 |
| Labeled Recall@5 | 0.850 | 0.850 |
| MRR@5 | 0.750 | 0.775 |

These are development-set diagnostics, **not estimates of general retrieval accuracy**: the set is small, some questions were developed after inspecting passages, and supporting-chunk labels are incomplete.

Initial manually inspected answer-generation examples are documented in [`eval/answer_evaluation.md`](eval/answer_evaluation.md):

| Example | Observation |
| --- | --- |
| Improving motor efficiency | Cited a retrieved page but overgeneralized qualified statements about motor design and enclosure. |
| Operating below 40% full load | Answer about low efficiency and poor power factor was supported by a retrieved passage on PDF page 27. |
| Facility Wi-Fi password | Abstained with an insufficient-documentation response instead of inventing a password. |

These three examples illustrate behavior; they do not establish an answer-accuracy or abstention percentage.

## Automated tests

```powershell
python -m pytest -q
```

**Latest reported local run:** 65 passed, 1 third-party deprecation warning (September 2026). The tests cover document processing, retrieval, persistence, prompt construction, client behavior, citation-reference validation, RAG orchestration, CLI, and API. The automated tests use mocks/synthetic fixtures where appropriate; they do not establish that the LLM's answers are always factual.

## Repository layout

```text
app/       PDF ingestion, preprocessing, chunking, embeddings, FAISS,
           prompting, LLM clients, RAG pipeline, citation checks, API
scripts/   Index building, retrieval, evaluation, prompt preview, Q&A CLI
tests/     Automated unit and API tests
eval/      Retrieval questions and manual answer-evaluation notes
data/      Local raw PDF and generated FAISS artifacts (ignored by Git)
```

## Limitations and next steps

- Only one configured, text-based sample PDF is indexed by the current script; scanned PDFs require OCR, which is not implemented.
- PDF extraction may introduce broken words, lose table structure, or split sentences at fixed-size chunk boundaries.
- Top-k vector retrieval can return irrelevant passages, including bibliography content. There is no calibrated out-of-domain similarity threshold.
- The LLM can overgeneralize, omit citations, or cite valid pages that do not fully support its claims. Citation validation checks *references*, not claim-level faithfulness.
- Abstention recognition currently relies on one exact response string, so alternate refusal phrasing may not be recognized.
- FastAPI caches local resources after first use; `/health` is not a dependency-readiness check. The local model requires sufficient system memory and may be slow on some computers.
- Dependency versions are not yet pinned and GitHub Actions CI, containerization, a frontend, multi-document ingestion, and broader evaluation are future improvements—not current features.

## Data and privacy

The sample sourcebook is attributed above and is not committed to this repository. `data/raw/`, `data/index/`, virtual environments, and `.env` files are ignored. Do not commit proprietary manuals, credentials, or personally identifiable information. The documented Ollama answer-generation path runs locally after setup, while initial installation/model downloads use network access.

## Project status

**Working prototype:** PDF → embeddings → FAISS retrieval → local LLM → citation report, available through CLI and FastAPI. The next release tasks are reproducible setup verification, dependency pinning/compatibility testing, GitHub Actions CI, and a short demo.
