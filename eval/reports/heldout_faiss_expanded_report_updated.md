# Held-out RAG Evaluation — FAISS + Adjacent-Chunk Expansion

**Run:** 27 September 2026 · **Dataset:** 18 held-out questions (15 answerable, 3 unanswerable) · **Source:** one industrial motor handbook · **Index:** 524 passages
**Configuration (frozen before evaluation):** all-MiniLM-L6-v2, FAISS top 5, up to 10 context passages via same-page adjacent-chunk expansion, local Ollama `llama3.2:3b`.
**Evaluation protocol:** one held-out generation run after freezing the configuration, with no tuning on test questions. The SHA-256 digest of the test dataset was compared with `eval/test_questions.sha256` in the project environment before evaluation and matched (`True`).

## Retrieval metrics

| Metric | Result |
|---|---:|
| Strict labeled Hit@5 (15 answerable questions) | 12/15 = 80.00% |
| Macro labeled Recall@5 | 0.7667 |
| MRR@5 | 0.6556 |
| Labeled evidence within expanded context (up to 10) | 13/15 = 86.67% |

The top-five labeled-evidence misses are Q043, Q049, and Q050. Expansion recovers Q043; Q049 and Q050 remain misses. Q043 contains relevant equivalent information on the cited PDF page 20, even though its specific gold chunk is on page 42; these strict metrics are conservative where annotations omit equivalent evidence.

## Generation and citation metrics

| Metric | Result |
|---|---:|
| Answerable questions receiving an answer | 13/15 |
| False abstentions on answerable questions | 2/15 (Q049, Q050) |
| Correct abstentions on unanswerable questions | 3/3 |
| Formally valid citation references in generated answers | 8/13 = 61.54% |

The citation validator checks reference syntax and whether the referenced page was supplied to the model; it **does not** verify that a page supports each generated claim. Citation failures: Q044 (malformed), Q047, Q051, Q055, Q056 (missing).

## Preliminary manual answer review

This is a qualitative first pass against the reference answers and retrieved passage excerpts, not an independent human-adjudicated accuracy benchmark. A “complete” label requires the expected central elements without a material factual error; “partial” denotes omissions/ambiguous or misleading additions; “incorrect” denotes a material error or false abstention.

- Complete: 8/15; partial: 4/15; incorrect: 3/15.

| Question | Review | Reason |
|---|---|---|
| Q043 | partial | Omitted that some DC motors can operate at zero RPM; relevant alternative evidence on p. 20 was retrieved. |
| Q044 | complete | Correct distinction and applications; citation formatting malformed. |
| Q045 | incorrect | Incorrectly attributes both voltage and enclosure selection to environmental conditions and calls voltage categories enclosure types; motor voltage should match available plant supply. |
| Q046 | partial | Both approaches and principal drawbacks stated; adds misleading claim about saving energy as a drawback. |
| Q047 | complete | All requested winding characteristics and wire supply information; no citation. |
| Q048 | complete | Correct 10-to-1 equivalence; cited retrieved passage. |
| Q049 | incorrect | Incorrect abstention: labeled proposal evidence not present in top-five or expanded context. |
| Q050 | incorrect | Incorrect abstention: labeled three-category passage not present in top-five or expanded context. |
| Q051 | complete | Correctly lists all five voltage-deviation factors; no citation. |
| Q052 | complete | Correct distinction between battery-based and flywheel-based UPS; cited passage. |
| Q053 | complete | Includes all three requested energy/environmental metrics; cited passage. |
| Q054 | complete | Correct earnings-increase times P/E relationship; cited passage. |
| Q055 | partial | Correct measurements and general formula, but says electricity costs rather than the required electricity rate; no citation. |
| Q056 | partial | Correct industry and general purpose, but omits specific what-if analysis capabilities in reference; no citation. |
| Q057 | complete | Correct regression-based weather/production normalization; cited passage. |
| Q058 | abstained correctly | No facility-specific sag timestamp in source. |
| Q059 | abstained correctly | No facility-specific invoice in source. |
| Q060 | abstained correctly | No facility-specific future inspection schedule in source. |

## Interpretation and limitations

- The held-out strict Hit@5 and expanded evidence coverage are below development-set levels (dev FAISS Hit@5 33/35; dev FAISS+expansion 34/35). The held-out sample has only 15 answerable questions from the same source PDF; do not infer broad deployment performance.
- Retrieval misses for Q049/Q050 prevent the model from accessing labeled evidence. Q045 demonstrates factual conflation despite retrieving its labeled source. Q044 demonstrates citation formatting failure despite a substantively correct answer.
- Some retrieved passages are tables of contents or other low-signal text. Generalized TOC filtering, structured table parsing, reranking and claim-level citation verification remain future work rather than part of the frozen evaluation.
- This is a local, single-document portfolio benchmark, not a production safety or correctness certification.

## Reproduction and artifacts

```powershell
python -m scripts.evaluate_answers_batch --dataset test --all --confirm-heldout --retrieval-mode faiss_expanded --top-k 5 --max-context-chunks 10 --output eval/runs/test_ollama_faiss_expanded_full.json
python -m pytest -q
(Get-FileHash .\eval\test_questions.json -Algorithm SHA256).Hash -eq (Get-Content .\eval\test_questions.sha256).Trim()
```

**Metadata correction:** The saved test-run JSON originally contained a literal `"dataset": "args.dataset"` field. It was corrected locally to `"dataset": "test"` without regenerating answers. The evaluator source should use `"dataset": args.dataset` to label future runs correctly. This reporting-only change does not alter evaluation outcomes.
