# PolyMentor — System Architecture

> Full design breakdown for developers, contributors, and integrators.

---

## Overview

PolyMentor is a modular AI mentoring system built around a five-stage pipeline: data ingestion, feature extraction, model inference, reasoning, and output synthesis. Every stage is independently replaceable. The single public entrypoint is `src/inference/pipeline.py`; all internal components are encapsulated behind it.

The system is designed to answer three questions about any submitted code snippet:

1. **What is wrong?** — Error type, location, and confidence score.
2. **Why is it wrong?** — A human-readable explanation tied to a programming concept.
3. **How should the learner think about it?** — A progressive, step-by-step hint calibrated to the user's skill level.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Input: Source Code                         │
│              (C++ / Python / JavaScript / Java)               │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   AST + Tokenization    │
              │  Tree-sitter parsing    │
              │  Code embeddings        │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Error Detection Model │
              │   CodeBERT classifier   │
              │   Multi-label output    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Reasoning Engine      │
              │  Error classification   │
              │  Concept mapping        │
              │  Difficulty scoring     │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
┌────────▼───────┐ ┌───────▼──────┐ ┌────────▼────────┐
│  Explanation   │ │     Hint     │ │  Quality Score  │
│  Generator     │ │   System     │ │  + Suggestions  │
│  (LLM / FT)    │ │  Step-by-step│ │                 │
└────────┬───────┘ └───────┬──────┘ └────────┬────────┘
         └─────────────────┼──────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Final Output        │
              │  · Error type & location│
              │  · Why it happened      │
              │  · Step-by-step hint    │
              │  · Concept taught       │
              │  · Quality score        │
              └─────────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Data Pipeline (`src/data_pipeline/`)

| Module | Role |
|---|---|
| `collector.py` | Scrapes and ingests raw code samples and labeled error datasets |
| `cleaner.py` | Normalises whitespace, removes noise, strips non-code content |
| `tokenizer.py` | Language-aware tokenisation — delegates to per-language config in `configs/language_config.yaml` |
| `dataset_builder.py` | Produces train / val / test splits as `data/processed/*.json` |

All raw material lives under `data/raw/` in three sub-collections: multi-language code samples, labeled buggy examples, and problem–solution pairs. The pipeline normalises all sources into a single schema before writing processed files.

---

### Stage 2 — Feature Extraction (`src/features/`)

| Module | Role |
|---|---|
| `ast_parser.py` | Uses Tree-sitter to generate language-agnostic ASTs from source code |
| `syntax_tree_builder.py` | Converts raw ASTs into structured, model-consumable tree representations |
| `code_embeddings.py` | Generates dense vector representations via `microsoft/codebert-base` |

Tree-sitter is chosen over language-specific parsers because it supports incremental, error-tolerant parsing across all four target languages with a unified API. CodeBERT embeddings carry semantic information about code patterns that token-level features cannot capture.

---

### Stage 3 — Model Layer (`src/models/`)

| Module | Role |
|---|---|
| `error_detector.py` | Multi-label CodeBERT classifier — one snippet may carry multiple error types simultaneously |
| `explanation_model.py` | Fine-tuned seq2seq model (CodeT5 or LLaMA) for natural-language explanations |
| `hint_generator.py` | Produces ordered, progressive hint sequences without revealing the solution |
| `model_factory.py` | Central loader — resolves model path, handles versioning, routes inference requests |

All saved checkpoints live in `models_saved/`. The factory loads the appropriate model based on task type and configuration, so callers never reference model files directly.

---

### Stage 4 — Reasoning Engine (`src/reasoning_engine/`)

This is the core intelligence layer. It takes raw model outputs and transforms them into pedagogically structured responses.

| Module | Role |
|---|---|
| `error_classifier.py` | Maps detected error types to programming concepts (e.g. off-by-one → loop indexing → iteration fundamentals) |
| `explanation_generator.py` | Composes the final explanation from error type, concept, and user skill level |
| `hint_system.py` | Builds the step-by-step hint sequence, calibrating depth to the learner's level |
| `feedback_scorer.py` | Scores submitted code on readability, complexity, and clean-code criteria |

The concept mapping in `error_classifier.py` is what differentiates PolyMentor from a standard linter. Instead of simply returning "SyntaxError at line 4", it links the error to the concept the learner needs to understand, and that concept drives both the explanation depth and the hint strategy.

---

### Stage 5 — Inference (`src/inference/`)

| Module | Role |
|---|---|
| `pipeline.py` | **Public API entrypoint.** Orchestrates all stages end-to-end. |
| `predict.py` | Wraps model forward passes; handles batching and device placement |
| `explain.py` | Calls the explanation model and formats output for the response schema |
| `tutor_mode.py` | Manages stateful, multi-turn interactive tutoring sessions |

External callers (the Polycode platform, the FastAPI backend, tests) interact exclusively with `pipeline.py`. All other modules in this package are implementation details.

---

## Configuration

Three YAML files control system behaviour:

| File | Controls |
|---|---|
| `configs/model_config.yaml` | Model architecture, hidden size, number of labels, dropout |
| `configs/training_config.yaml` | Batch size, learning rate schedule, epochs, checkpointing |
| `configs/language_config.yaml` | Per-language tokeniser paths, max sequence lengths, special tokens |

Copy the relevant file to a `.local.yaml` variant before editing — local configs are gitignored and take precedence over defaults.

---

## Output Schema

Every call to `pipeline.analyze()` returns a structured result object:

```python
result.error_type        # str  — e.g. "SyntaxError: assignment in condition"
result.error_location    # int  — line number
result.explanation       # str  — plain-English explanation of what went wrong and why
result.hint              # str  — first hint step (additional steps on demand)
result.concept_taught    # str  — e.g. "Comparison Operators: == vs ="
result.quality_score     # int  — 0–100 composite code quality score
result.suggestions       # list[str] — concrete improvement recommendations
```

---

## Deployment

The inference pipeline is served via a FastAPI + Uvicorn application. The recommended deployment target is Docker on AWS. The Docker image packages the model checkpoints, configs, and the FastAPI app together. The API exposes a single `/analyze` POST endpoint that accepts a code string, language, and learner level, and returns the output schema above as JSON.

---

## Relationship to PolyGuard

PolyMentor and PolyGuard share the same CodeBERT backbone and Tree-sitter AST infrastructure. PolyGuard focuses on security vulnerability detection and secure fix generation; PolyMentor focuses on learning-oriented error explanation and tutoring. The two systems are designed to be composed: PolyGuard flags a security issue, PolyMentor explains the underlying concept to the developer. Together they form the AI developer intelligence core of the Polycode platform.

See [`future_polycode_integration.md`](future_polycode_integration.md) for the integration plan.
