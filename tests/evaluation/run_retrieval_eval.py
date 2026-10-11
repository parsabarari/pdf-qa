import argparse
import json
import logging
from pathlib import Path

from app.core.config import get_settings
from app.services.embeddings import embed_query
from app.services.vector_store import get_vector_store_for_path


logger = logging.getLogger(__name__)

EVALUATION_FILE = Path(__file__).with_name("retrieval_cases.json")


def load_evaluation_cases() -> list[dict]:
    """Load evaluation questions and expected source pages."""
    with EVALUATION_FILE.open("r", encoding="utf-8-sig") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must contain a non-empty JSON list")

    for case in cases:
        if not all(key in case for key in ("case_id", "question", "expected_pages")):
            raise ValueError(f"Invalid evaluation case: {case}")

        if not case["expected_pages"]:
            raise ValueError(
                f"Case {case['case_id']} must have at least one expected page"
            )

    return cases


def evaluate_case(
    case: dict,
    document_id: str,
    settings,
    store,
) -> dict:
    """Retrieve chunks for one question and calculate page-level Recall@k."""
    question = case["question"]
    expected_pages = set(case["expected_pages"])

    query_embedding = embed_query(question, settings)

    retrieved_chunks = store.query(
        document_id=document_id,
        embedding=query_embedding,
        top_k=settings.top_k,
    )

    retrieved_pages = {chunk.page for chunk in retrieved_chunks}
    relevant_pages = expected_pages & retrieved_pages

    recall = len(relevant_pages) / len(expected_pages)

    filtered_chunks = [
        chunk
        for chunk in retrieved_chunks
        if chunk.score >= settings.min_relevance_score
    ]

    filtered_pages = {chunk.page for chunk in filtered_chunks}
    filtered_relevant_pages = expected_pages & filtered_pages

    filtered_recall = (
        len(filtered_relevant_pages) / len(expected_pages)
    )

    reciprocal_rank = 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.page in expected_pages:
            reciprocal_rank = 1.0 / rank
            break

    filtered_reciprocal_rank = 0.0

    for rank, chunk in enumerate(filtered_chunks, start=1):
        if chunk.page in expected_pages:
            filtered_reciprocal_rank = 1.0 / rank
            break

    return {
        "case_id": case["case_id"],
        "expected_pages": sorted(expected_pages),
        "retrieved_pages": sorted(retrieved_pages),
        "recall": recall,
        "num_retrieved_chunks": len(retrieved_chunks),
        "chunk_details": [
            {
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
            }
            for chunk in retrieved_chunks
        ],
        "filtered_pages": sorted(filtered_pages),
        "filtered_recall": filtered_recall,
        "reciprocal_rank": reciprocal_rank,
        "filtered_reciprocal_rank": filtered_reciprocal_rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate page-level retrieval recall for a PDF document"
    )
    parser.add_argument(
        "--document-id",
        required=True,
        help="ID of the previously uploaded PDF document",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    store = get_vector_store_for_path(settings.chroma_path)
    cases = load_evaluation_cases()

    results = []

    for case in cases:
        result = evaluate_case(
            case=case,
            document_id=args.document_id,
            settings=settings,
            store=store,
        )
        results.append(result)

        print(f"\nCase: {result['case_id']}")
        print(f"Expected pages:  {result['expected_pages']}")
        print(f"Retrieved pages: {result['retrieved_pages']}")
        print(
            f"Recall@{settings.top_k}: "
            f"{result['recall']:.2f}"
        )
        print(f"Retrieved chunks: {result['num_retrieved_chunks']}")

        for chunk in result["chunk_details"]:
            print(
                f"  page={chunk['page']}, "
                f"chunk={chunk['chunk_index']}, "
                f"score={chunk['score']:.4f}"
            )

        print(
            f"Recall after threshold "
            f"({settings.min_relevance_score}): "
            f"{result['filtered_recall']:.2f}"
        )
        print(f"Pages after threshold: {result['filtered_pages']}")

    mean_reciprocal_rank = sum(
        result["reciprocal_rank"] for result in results
    ) / len(results)

    mean_filtered_reciprocal_rank = sum(
        result["filtered_reciprocal_rank"] for result in results
    ) / len(results)

    mean_filtered_recall = sum(
        result["filtered_recall"] for result in results
    ) / len(results)

    print(
        f"Mean Recall after threshold "
        f"({settings.min_relevance_score}): "
        f"{mean_filtered_recall:.4f}"
    )

    mean_recall = sum(result["recall"] for result in results) / len(results)

    print("\n" + "=" * 40)
    print(f"Evaluation cases: {len(results)}")
    print(f"Mean Recall@{settings.top_k}: {mean_recall:.4f}")
    print("=" * 40)
    print(f"Mean MRR@{settings.top_k}: {mean_reciprocal_rank:.4f}")
    print(
        f"Mean MRR after threshold "
        f"({settings.min_relevance_score}): "
        f"{mean_filtered_reciprocal_rank:.4f}"
    )


if __name__ == "__main__":
    main()
