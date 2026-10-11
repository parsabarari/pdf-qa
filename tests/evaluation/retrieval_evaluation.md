## Retrieval Evaluation

This project includes a lightweight evaluation script for measuring how effectively the retrieval component finds relevant pages in a PDF document.

The evaluation focuses on **retrieval quality**, not answer generation. It helps determine whether the vector search returns chunks from the pages expected to contain the answer, and whether those chunks appear near the top of the ranked results.

### How It Works

The evaluation pipeline follows these steps:

1. **Prepare evaluation cases:** Define questions and their expected relevant PDF pages in `tests/evaluation/retrieval_cases.json`.
2. **Embed each question:** Generate an embedding for every evaluation question using the configured embedding model.
3. **Retrieve relevant chunks:** Query ChromaDB using the question embedding and retrieve up to `TOP_K` chunks.
4. **Evaluate retrieved pages:** Compare the page numbers of retrieved chunks with the expected page numbers defined in the evaluation dataset.
5. **Apply the relevance threshold:** Filter retrieved chunks using `MIN_RELEVANCE_SCORE` to evaluate whether the relevance threshold removes useful results.
6. **Calculate metrics:** Report Recall@K and Mean Reciprocal Rank (MRR), both before and after relevance filtering.

### Evaluation Dataset

Evaluation cases are stored in `tests/evaluation/retrieval_cases.json`.

Each case contains a unique identifier, a question, and a list of expected relevant PDF pages.

Example:

```json
{
  "case_id": "qa_001",
  "question": "What does the time complexity of an algorithm estimate?",
  "expected_pages": [27]
}
```

The expected pages act as the ground-truth labels for evaluating retrieval results. These labels should be reviewed and expanded as the evaluation dataset grows.

### Evaluation Metrics

#### Recall@K

Recall@K measures whether the retrieved top-K chunks include the expected relevant pages.

For each test case:

`Recall@K = relevant pages retrieved / expected relevant pages`

A score of `1.0` means all expected relevant pages were represented in the retrieved results. A score of `0.0` means none were retrieved.

The evaluation also calculates Recall after applying `MIN_RELEVANCE_SCORE`, allowing the results to show whether threshold filtering removes relevant content.

#### Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant result appears in the ranking.

`Reciprocal Rank = 1 / rank of first relevant result`

If the first relevant result is ranked first, the reciprocal rank is `1.0`. If it appears fourth, the reciprocal rank is `0.25`. If no relevant result is found, the score is `0`.

MRR is calculated across all evaluation cases to measure how consistently relevant content appears near the top of the results.

### Relevance Threshold

The application uses `MIN_RELEVANCE_SCORE` to filter retrieved chunks before passing context to the answer-generation stage.

ChromaDB is configured to use cosine distance. The vector store converts the returned distance into a similarity-style score:

`score = 1 - cosine_distance`

Higher scores indicate greater similarity under this scoring convention. The evaluation uses the same threshold as the application so that filtered retrieval can be assessed under the actual configuration.

The current threshold is:

```env
MIN_RELEVANCE_SCORE=0.25
```

This is a configurable starting point, not a universally optimal threshold. Its effectiveness should be evaluated against representative questions and relevant-page labels.

### Running the Evaluation

Start the application and ensure the target PDF has already been uploaded and processed. Then run the evaluation script from the project root:

```bash
python tests/evaluation/run_retrieval_eval.py \
  --document-id YOUR_DOCUMENT_ID
```

Replace `YOUR_DOCUMENT_ID` with the ID of the processed document in ChromaDB.

The script reads the evaluation cases, runs retrieval for each question, displays the retrieved chunks and their scores, and reports aggregate metrics. The number of retrieved chunks is controlled by `TOP_K`.

### Baseline Results

The initial evaluation was run against the *Competitive Programmer’s Handbook* PDF using five manually defined questions.

| Metric | Baseline result |
|---|---:|
| Number of evaluation cases | 5 |
| Recall@4 | 1.00 |
| Recall@4 after threshold filtering | 1.00 |
| Mean Reciprocal Rank (MRR@4) | 0.80 |
| MRR@4 after threshold filtering | 0.80 |

**Interpretation:** All five test cases retrieved their expected relevant pages within the top four results. Applying the current relevance threshold did not reduce measured recall or MRR on this test set.

These results provide an initial baseline for the retrieval component. Because the dataset contains only five questions, they should not be interpreted as a guarantee of retrieval quality across all documents or question types.

### Limitations

- The evaluation measures page-level retrieval relevance, not the factual correctness of generated answers.
- Results depend on the quality and coverage of the manually assigned expected pages.
- A small evaluation dataset may not represent more complex questions or different PDFs.
- Recall@K and MRR do not directly measure answer faithfulness, citation accuracy, latency, or cost.

The evaluation dataset can be expanded over time to track retrieval quality and detect regressions when changing embedding models, chunking settings, `TOP_K`, or the relevance threshold.
