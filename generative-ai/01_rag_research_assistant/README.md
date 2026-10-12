# RAG Research Assistant

A retrieval-augmented generation case study over a curated corpus of foundational papers on Transformers, information retrieval, and RAG.

The project focuses on the part of RAG that is often hidden behind a framework call: **retrieval quality**. Dense retrieval, BM25, hybrid fusion, reranking, and chunk size are evaluated independently before the final generator is used.

## Architecture

```text
arXiv PDFs
    ↓
PyMuPDF + metadata
    ↓
700-token chunks / 75 overlap
    ↓
┌────────────────────────────┐
│ BGE-small dense retrieval  │
│ Qdrant BM25 sparse search  │
└─────────────┬──────────────┘
              ↓
          Qdrant RRF
       20 candidates
              ↓
 MiniLM cross-encoder reranker
          top 6
              ↓
     ±1 context expansion
              ↓
       GPT-5.6 Luna
              ↓
 grounded answer + citations
```

## Corpus

The corpus contains 11 papers covering:

- Transformer architecture
- BERT
- Sentence-BERT
- Dense Passage Retrieval
- ColBERT
- Retrieval-Augmented Generation
- Fusion-in-Decoder
- HyDE
- Lost in the Middle
- Self-RAG
- RAFT

PDFs are downloaded from arXiv at runtime and are not committed to the repository.

## Retrieval experiments

### Dense vs. BM25 vs. hybrid

On the 20-query development benchmark:

| Metric | Dense | BM25 | Hybrid RRF |
|---|---:|---:|---:|
| MRR@10 | 0.9375 | 0.8958 | **1.0000** |
| Hit@1 | 0.900 | 0.850 | **1.000** |
| Recall@1 | 0.725 | 0.675 | **0.800** |
| Recall@5 | 0.850 | 0.850 | **0.875** |
| Recall@10 | **0.975** | 0.950 | 0.950 |

Dense retrieval performs better on semantic reformulations, while BM25 is particularly effective for exact names and rare technical terms. RRF combines both signals and improves early ranking, although it does not universally improve deep recall.

### Cross-encoder reranking

Using the same operational budget as the final system:

| Metric | Hybrid RRF | Hybrid + reranker |
|---|---:|---:|
| MRR@6 | **1.0000** | 0.9625 |
| Hit@1 | **1.000** | 0.950 |
| Recall@3 | 0.850 | **0.900** |
| Recall@5 | 0.875 | **0.925** |
| Recall@6 | 0.875 | **0.925** |

The reranker sacrifices a small amount of first-result ranking quality but improves later source coverage, which is more useful for multi-source generation.

### Chunk-size ablation

| Chunk size | Chunks | Hit@3 | Recall@5 | Recall@6 | MRR@6 |
|---:|---:|---:|---:|---:|---:|
| 250 | 975 | **1.00** | 0.900 | 0.925 | **1.0000** |
| 450 | 500 | 0.95 | **0.925** | 0.925 | 0.9625 |
| **700** | **327** | **1.00** | **0.925** | 0.925 | **1.0000** |

The 700-token configuration is selected by a predefined rule: maximize Recall@5, then MRR@6, then prefer the smaller index.

## Locked retrieval holdout

After the retrieval architecture and chunk size are fixed, the final system is evaluated on 12 new source-level queries.

| Metric | Result |
|---|---:|
| MRR@6 | **1.0000** |
| Hit@1 | **1.0000** |
| Recall@1 | 0.9167 |
| Recall@3 | 0.9583 |
| Recall@5 | **0.9583** |
| Recall@6 | **0.9583** |

Every query retrieves at least one expected source at rank 1.

The remaining recall error is a multi-source DPR/ColBERT comparison: ColBERT is retrieved, but DPR does not survive the final top-6 cutoff.

This illustrates an important limitation of source-level retrieval evaluation: a perfect first relevant hit does not guarantee that every source needed for a comparison question reaches the generator.

## Grounded generation

Five representative holdout questions are passed through the full RAG system.

Single-source questions are answered from the expected source with explicit metadata-based citations.

For the DPR/ColBERT failure case, the generator explicitly states that DPR is not present in the supplied context instead of presenting unsupported details as retrieved evidence. A Self-RAG/RAFT comparison successfully retrieves both sources and produces a grounded multi-source answer.

## Abstention

Five deliberately out-of-corpus questions cover LoRA, FlashAttention implementation details, Mamba, CLIP, and GPT-4 training details.

Manual review finds **5 / 5 correct abstentions**. Some answers provide high-level information that is actually present in the retrieved corpus, but they explicitly refuse to infer the unsupported detail requested by the question.

A simple phrase-based heuristic only detects 40% of those refusals, demonstrating why semantic behavior should not be evaluated with a small keyword list.

## Cost

The retrieval stack runs locally.

For the final qualitative checks:

- five grounded answers: **$0.01086**
- five out-of-corpus checks: **$0.01069**
- total: **$0.02155**

## Main conclusions

- Dense and lexical retrieval have complementary failure modes.
- Hybrid RRF substantially improves top-rank reliability.
- Fusion can still hurt multi-source recall.
- Cross-encoder reranking improves evidence coverage even when it slightly reduces rank-1 metrics.
- Chunk size should be evaluated rather than selected arbitrarily.
- High source-level retrieval metrics do not guarantee complete evidence coverage.
- Grounded refusal is preferable to filling missing context with parametric knowledge.
- A strong laptop-scale RAG pipeline can keep retrieval local and use a hosted LLM for only a few cents.

## Limitations

- The corpus is small and curated.
- Retrieval labels identify papers rather than exact answer-bearing chunks.
- The development and holdout benchmarks are small.
- Multi-source coverage remains the main retrieval weakness.
- Generation quality is reviewed manually rather than with an automated LLM judge.
- PDF extraction is text-first and does not semantically reconstruct figures or tables.
- Local embedding and reranking models prioritize reproducibility over maximum possible retrieval quality.

## Reproducibility

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add an OpenAI API key:

```text
OPENAI_API_KEY=...
```

Run:

```text
rag_research_assistant_case_study.ipynb
```

from top to bottom.

The notebook downloads the PDFs automatically and builds the local Qdrant index.

## Project structure

```text
01_rag_research_assistant/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── papers_manifest.json
├── rag_research_assistant_case_study.ipynb
└── evaluation/
    ├── retrieval_benchmark.json
    └── unanswerable_queries.json
```
