# RAG Answer Evaluation

## Objective

Evaluate whether the Industrial Knowledge RAG Assistant:
- Answers questions using retrieved document evidence.
- Provides valid source citations.
- Preserves technical qualifications.
- Abstains when the documentation is insufficient.

## Evaluation setup

- Document: Industrial motor manual
- Embedding model: all-MiniLM-L6-v2
- Vector database: FAISS
- LLM: Llama 3.2 3B via Ollama
- Retrieval: Top 5 chunks
- Generation: Local inference

## Evaluation criteria

For each question, record:

1. Answerable: Is the answer present in the retrieved evidence?
2. Citation validity: Does every bracketed citation reference a retrieved page?
3. Grounding: Are all factual claims supported by the cited passages?
4. Relevance: Does the response directly address the question?
5. Abstention: Does the system decline unsupported questions?

## Initial answer evaluation

**Setup:** Single industrial motor manual, `all-MiniLM-L6-v2` embeddings, FAISS retrieval with `top_k=5`, and Llama 3.2 3B through local Ollama.

| ID | Question                                                   | Observed result                                                                                                                | Assessment                                                    |
| -- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| Q1 | How can motor efficiency be improved?                      | Generated an answer with a valid page 43 citation, but presented qualified efficiency observations as general recommendations. | Citation reference valid; answer grounding needs improvement. |
| Q2 | What happens when a motor operates below 40% of full load? | Reported poor power factor and low efficiency, citing page 27. The retrieved passage supports this statement.                  | Supported answer with valid citation.                         |
| Q3 | What is the Wi-Fi password for the facility?               | Returned the insufficient-documentation message. API status was `abstained`, with citation validation marked not applicable.   | Appropriate abstention in this test.                          |

### Findings

* The API successfully returns generated answers, retrieved evidence, answer status, and citation-reference validation.
* The citation validator verifies that references point to retrieved source/page combinations. It does not verify factual support for individual claims.
* The motor-efficiency example demonstrates that an answer can contain a valid citation while still overstating the source.
* The low-load example demonstrates that a relevant passage can support an answer even when the expected supporting page was not retrieved.
* The unrelated Wi-Fi question demonstrates successful abstention in one observed run.

### Limitations

These three examples are a small, manually inspected development set. They are not representative enough to establish overall accuracy, reliability, or abstention rates. Additional questions and repeated runs are needed for a broader assessment.
