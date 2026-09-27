import pytest

from app import rag_pipeline


def make_chunk(page, number, text):
    return {
        "chunk_id": f"motor_manual.pdf_p{page}_c{number}",
        "source": "motor_manual.pdf",
        "page": page,
        "text": text,
    }


@pytest.fixture
def sample_chunks():
    ranked = [
        make_chunk(5, 10, "Inspection table contents."),
        make_chunk(39, 0, "Common inspection tasks table."),
        make_chunk(32, 2, "Other motor information."),
        make_chunk(34, 1, "Other inspection information."),
        make_chunk(5, 9, "Sourcebook contents."),
    ]

    annual = make_chunk(
        39,
        1,
        "Annually: regrease antifriction bearings, "
        "check air gap, check bearing clearances and "
        "clean undercut slots in the commutator.",
    )

    return ranked, ranked + [annual], annual


def test_faiss_remains_default(monkeypatch, sample_chunks):
    ranked, corpus, _ = sample_chunks

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: ranked[:1],
    )

    def unexpected_hybrid(**kwargs):
        raise AssertionError("Hybrid should not be called.")

    monkeypatch.setattr(
        rag_pipeline,
        "search_hybrid_index",
        unexpected_hybrid,
    )

    context = rag_pipeline.prepare_rag_context(
        question="What are common inspection tasks?",
        model=None,
        index=None,
        chunks=corpus,
    )

    assert len(context["sources"]) == 1
    assert "retrieved_sources" not in context


def test_hybrid_recovers_adjacent_chunk(
    monkeypatch, sample_chunks
):
    ranked, corpus, annual = sample_chunks

    monkeypatch.setattr(
        rag_pipeline,
        "search_hybrid_index",
        lambda **kwargs: ranked,
    )

    context = rag_pipeline.prepare_rag_context(
        question="What are the annual maintenance tasks?",
        model=None,
        index=None,
        chunks=corpus,
        retrieval_mode="hybrid",
        bm25_retriever=object(),
    )

    retrieved_ids = {
        item["chunk_id"]
        for item in context["retrieved_sources"]
    }

    expanded_ids = {
        item["chunk_id"]
        for item in context["sources"]
    }

    assert len(context["retrieved_sources"]) == 5
    assert annual["chunk_id"] not in retrieved_ids
    assert annual["chunk_id"] in expanded_ids
    assert annual["text"] in context["prompt"]


def test_generation_validates_expanded_evidence(
    monkeypatch, sample_chunks
):
    ranked, corpus, annual = sample_chunks

    monkeypatch.setattr(
        rag_pipeline,
        "search_hybrid_index",
        lambda **kwargs: ranked,
    )

    def fake_validate(answer, retrieved_sources):
        ids = {
            item["chunk_id"]
            for item in retrieved_sources
        }

        assert annual["chunk_id"] in ids

        return {"all_valid": True}

    monkeypatch.setattr(
        rag_pipeline,
        "validate_citations",
        fake_validate,
    )

    class FakeLLM:
        def generate(self, prompt):
            assert annual["text"] in prompt

            return (
                "Regrease antifriction bearings. "
                "[motor_manual.pdf, p. 39]"
            )

    result = rag_pipeline.generate_rag_answer(
        question="What are the annual maintenance tasks?",
        model=None,
        index=None,
        chunks=corpus,
        llm=FakeLLM(),
        retrieval_mode="hybrid",
        bm25_retriever=object(),
    )

    assert result["answer_status"] == "answered"
    assert len(result["retrieved_sources"]) == 5
    assert annual["chunk_id"] in {
        item["chunk_id"] for item in result["sources"]
    }
    assert result["citation_validation"] == {
        "all_valid": True
    }


@pytest.mark.parametrize(
    "mode, retriever, error",
    [
        ("invalid", None, "retrieval_mode"),
        ("hybrid", None, "BM25"),
    ],
)
def test_rejects_invalid_retrieval_configuration(
    mode, retriever, error
):
    with pytest.raises(ValueError, match=error):
        rag_pipeline.prepare_rag_context(
            question="Motor maintenance?",
            model=None,
            index=None,
            chunks=[],
            retrieval_mode=mode,
            bm25_retriever=retriever,
        )

def test_faiss_expansion_recovers_adjacent_chunk(
    monkeypatch, sample_chunks
):
    ranked, corpus, annual = sample_chunks

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: ranked,
    )

    context = rag_pipeline.prepare_rag_context(
        question="What are the annual maintenance tasks?",
        model=None,
        index=None,
        chunks=corpus,
        retrieval_mode="faiss_expanded",
        top_k=5,
        max_context_chunks=10,
    )

    retrieved_ids = {
        item["chunk_id"]
        for item in context["retrieved_sources"]
    }

    expanded_ids = {
        item["chunk_id"]
        for item in context["sources"]
    }

    assert annual["chunk_id"] not in retrieved_ids
    assert annual["chunk_id"] in expanded_ids
    assert annual["text"] in context["prompt"]
    assert len(context["retrieved_sources"]) == 5
    assert len(context["sources"]) <= 10
