import argparse
import os
from pathlib import Path

from app.embeddings import MODEL_NAME, load_embedding_model
from app.vector_store import load_faiss_index
from app.rag_pipeline import (
    prepare_rag_context,
    generate_rag_answer,
)
from app.llm_client import OpenAILLMClient, OllamaLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"


def main():

    parser = argparse.ArgumentParser(
        description="Ask questions about industrial documentation."
    )

    parser.add_argument(
        "question",
        help="Question to ask",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider (default: openai)",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Generate an answer instead of running a dry run",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider's default model",
    )

    args = parser.parse_args()

    # Validate arguments before loading the models.
    if not args.question.strip():
        parser.error("Question cannot be empty.")

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    # Only OpenAI requires the cloud API key.
    if (
        args.live
        and args.provider == "openai"
        and not os.getenv("OPENAI_API_KEY")
    ):
        parser.error(
            "OPENAI_API_KEY is required for OpenAI live mode."
        )

    print("Loading FAISS index...")

    index, chunks = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(f"Loaded vectors: {index.ntotal}")

    print("Loading embedding model...")

    embedding_model = load_embedding_model()

    # Dry-run mode: no LLM request.
    if not args.live:

        result = prepare_rag_context(
            question=args.question,
            model=embedding_model,
            index=index,
            chunks=chunks,
            top_k=args.top_k,
        )

        print("\nDRY RUN — No LLM request made.")

        print(
            f"Retrieved passages: {len(result['sources'])}"
        )

        print(
            "Prompt ready:",
            result["prompt"] is not None,
        )

    else:

        # Select the generation provider.
        if args.provider == "ollama":

            model_name = args.model or "llama3.2:3b"

            print(
                f"Generating answer locally using {model_name}..."
            )

            llm = OllamaLLMClient(
                model=model_name,
            )

        else:

            model_name = (
                args.model
                or os.getenv("RAG_LLM_MODEL", "gpt-5-mini")
            )

            print(
                f"Generating answer using OpenAI: {model_name}..."
            )

            llm = OpenAILLMClient(
                model=model_name,
            )

        result = generate_rag_answer(
            question=args.question,
            model=embedding_model,
            index=index,
            chunks=chunks,
            llm=llm,
            top_k=args.top_k,
        )

        print("\n" + "=" * 60)
        print("GENERATED ANSWER")
        print("=" * 60)

        print(result["answer"])

    # Display retrieved sources.
    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for rank, source in enumerate(
        result["sources"],
        start=1,
    ):

        print(
            f"{rank}. {source['source']} "
            f"| Page {source['page']} "
            f"| Score: {source['score']:.4f} "
            f"| Chunk: {source['chunk_id']}"
        )

    if args.live:

        report = result["citation_validation"]

        print("\n" + "=" * 60)
        print("CITATION VALIDATION")
        print("=" * 60)

        if args.live:
            report = result["citation_validation"]

            print("\n" + "=" * 60)
            print("CITATION VALIDATION")
            print("=" * 60)

            if report is None:
                if result["answer_status"] == "abstained":
                    print(
                        "Not applicable: the model abstained "
                        "because evidence was insufficient."
                    )
                else:
                    print(
                        "Not applicable: no document passages "
                        "were retrieved."
                    )
            else:
                print(f"Citations found: {report['has_citations']}")
                print(f"Reference validation passed: {report['is_valid']}")
                print("Valid citations:", report["valid_citations"])
                print("Invalid citations:", report["invalid_citations"])

            print(
                "\nNote: Reference validation does not verify "
                "that every answer claim is supported."
            )


if __name__ == "__main__":
    main()