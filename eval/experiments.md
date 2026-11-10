# Retrieval & Ranking Evaluation Experiments: Advanced Hybrid Engine

**Module**: Hybrid Information Retrieval & Evaluation  
**Author**: Srujan (Senior Information Retrieval & RAG Engineer)  
**System**: Railway Passenger Help Desk – GenAI Nexus Module 2  
**Date**: November 2026  

---

# Evaluation (Team 6)

## How we evaluated
- Test set: `eval/test_set.json`, 28 questions: the 10 sample questions from the pack, 13 questions whose answers we checked against the documents or the database, and 5 timetable questions (including unanswerable and harmful ones). Every document question is labelled with the document and page that contain the answer.
- Retrieval metrics: Hit@5, MRR@5, Recall@5, computed by `eval/evaluate.py` (a page within one page of the label counts as a match).
- End-to-end metrics (first 10 questions): answer accuracy, source correctness, abstention accuracy.

## Retrieval results (final index: chunk size 1500, overlap 150)
|          Configuration                | Hit@5  | MRR@5  | Recall@5 |
|---------------------------------------|--------|--------|----------|
| BM25 only                             | 1.0000 | 0.8393 | 0.9643   |
| Dense only                            | 1.0000 | 0.8810 | 1.0000   |
| Dense + BM25 (linear)                 | 1.0000 | 0.8929 | 1.0000   |
| Dense + BM25 + RRF                    | 1.0000 | 0.8571 | 0.9643   |
| + exact-code match                    | 1.0000 | 0.8571 | 0.9643   |
| + cross-encoder rerank (final system) | 1.0000 | 0.8214 | 1.0000   |

Findings: every configuration finds the right document and page in the top 5 (Hit@5 = 1.0), so Hit@5 does not separate them on this test set. MRR differs a little: the plain Dense + BM25 combination ranks the right chunk highest (0.8929), and the cross-encoder reranker did not improve MRR (0.8214). We kept the full pipeline because the answer step reads all five retrieved passages, and Hit@5 and Recall@5 stay at 1.0. With 28 questions, one chunk moving one place changes MRR by about 0.03 to 0.04, so these differences are small.

## Before and after: chunk size 500 -> 1500 (overlap 100 -> 150)
|     Configuration  |     Hit@5        |      MRR@5       |    Recall@5      |
|--------------------|------------------|------------------|------------------|
| BM25 only          | 0.9286 -> 1.0000 | 0.8214 -> 0.8393 | 0.8571 -> 0.9643 |
| Dense only         | 1.0000 -> 1.0000 | 0.8631 -> 0.8810 | 0.9643 -> 1.0000 |
| Dense + BM25       | 1.0000 -> 1.0000 | 0.8810 -> 0.8929 | 0.8929 -> 1.0000 |
| + RRF              | 1.0000 -> 1.0000 | 0.8810 -> 0.8571 | 0.8929 -> 0.9643 |
| + exact codes      | 1.0000 -> 1.0000 | 0.8810 -> 0.8571 | 0.8929 -> 0.9643 |
| Final (+ reranker) | 1.0000 -> 1.0000 | 0.8929 -> 0.8214 | 0.9643 -> 1.0000 |

Average Recall@5 over the six configurations rose from 0.911 to 0.982. MRR was mixed (up for three configurations, down for three). End-to-end results on the first 10 questions were 1.0 / 1.0 / 1.0 (answer accuracy / source correctness / abstention) in both runs, so the metrics could not separate the two settings. We made the change because of answer-level failures found by hand: with 500-character chunks the Vande Bharat rule was cut into three fragments, and two of the ten sample questions (S07, S08) were answered wrongly; with 1500-character chunks, table lead-in sentences and page-accurate sections all ten are correct.

| Other change | Effect |
|---|---|
| Installing `rank_bm25` | keyword search was silently off before |
| One retry when a SQL query returns 0 rows | stops of train 12002: "no matching records" -> correct stops and longest halt |
| Retry on transient Gemini errors | a 503 during evaluation was handled without failing |

## What we changed because of these results
1. 500-character chunks cut the Vande Bharat rule into three fragments, so the model hedged. Chunks are now 1500 characters (about 10% overlap), tables keep their lead-in sentence, and sections split at page boundaries so the page number is exact.
2. Station names are matched exact-name-first with a LIKE fallback, after a query for "Bhopal" returned nothing because the station is called "BHOPAL JN".
3. Answer rule: a rule written for one train type is not applied to another type.

## Known limits
- The test set is small (28 questions) and Hit@5 is saturated, so it cannot rank the retrieval methods well.
- End-to-end metrics cover only the first 10 questions; the other checks were manual.
- We do not compute a numeric routing confidence; routing uses keyword rules and an LLM router (see README).
- Document dates were copied from the pack's `SOURCES.md` into the ingestion config.
- The free Gemini key allows 15 requests per minute, so long runs wait and retry.
- Text extracted from some PDFs loses the space at a line break (for example "betweenseventy-two"); the numbers are not affected.