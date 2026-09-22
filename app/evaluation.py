def score_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    """
    Evaluate retrieval results for a single question.

    Returns:
        hit_rate, labeled_recall, reciprocal_rank
    """

    # Step 1: Validate inputs
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant_ids:
        raise ValueError("At least one relevant chunk is required.")

    if len(relevant_ids) != len(set(relevant_ids)):
        raise ValueError("Duplicate relevance labels are not allowed.")

    # Step 2: Consider only the top-k retrieved chunks
    top_k_ids = retrieved_ids[:k]

    relevant_set = set(relevant_ids)

    # Step 3: Find the relevant chunks retrieved
    found_relevant = set(top_k_ids) & relevant_set

    # Step 4: Calculate Hit Rate
    hit_rate = float(len(found_relevant) > 0)

    # Step 5: Calculate Labeled Recall
    labeled_recall = (
        len(found_relevant) / len(relevant_set)
    )

    # Step 6: Calculate Reciprocal Rank
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(top_k_ids, start=1):

        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "labeled_recall": labeled_recall,
        "reciprocal_rank": reciprocal_rank,
    }
def summarize_evaluation(
    evaluation_results: list[dict],
    k: int = 5,
) -> dict:
    """
    Calculate aggregate retrieval metrics across questions.

    Each input record contains:
        question_id
        retrieved_chunk_ids
        relevant_chunk_ids
    """

    if not evaluation_results:
        raise ValueError("Evaluation results cannot be empty.")

    if k <= 0:
        raise ValueError("k must be positive.")

    per_question = []

    # Evaluate every question individually.
    for item in evaluation_results:

        scores = score_retrieval(
            retrieved_ids=item["retrieved_chunk_ids"],
            relevant_ids=item["relevant_chunk_ids"],
            k=k,
        )

        per_question.append({
            "question_id": item["question_id"],
            **scores,
        })

    # Calculate averages across all questions.
    n = len(per_question)

    hit_rate = sum(
        item["hit_rate"]
        for item in per_question
    ) / n

    labeled_recall = sum(
        item["labeled_recall"]
        for item in per_question
    ) / n

    mrr = sum(
        item["reciprocal_rank"]
        for item in per_question
    ) / n

    return {
        "question_count": n,
        "k": k,
        "hit_rate_at_k": hit_rate,
        "labeled_recall_at_k": labeled_recall,
        "mrr_at_k": mrr,
        "per_question": per_question,
    }