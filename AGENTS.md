# AGENTS.md

## Goal

Refactor a legacy toxicity detection notebook into a clean, reusable inference pipeline.

Build a stable baseline first, then extend later into a policy-aware moderation system.

---

## Current Phase

Phase 1: Inference Pipeline Only

Focus on:

* Refactoring notebook logic into reusable modules
* Implementing a single comment prediction pipeline
* Loading model and embedding artifacts
* Returning `{label, score}` output

Do NOT:

* Add FastAPI
* Add RAG or retrieval logic
* Add vector database
* Add UI or frontend
* Replace the existing model
* Redesign the architecture

---

## Scope Rules

* Do not go beyond the requested scope.
* Implement only what is explicitly required for the current step.
* Work step-by-step, one feature at a time.
* Prefer the smallest working change over broad redesign.
* Do not refactor unrelated parts of the codebase.

---

## Core Code Rules

* Separate training, inference, and evaluation logic.
* Do not leave debug print statements in final code.
* Add docstrings only for public functions.
* Use short Korean docstrings.
* Each file must have a single responsibility.
* Prefer composable functions (preprocess -> embed -> predict).
* Avoid large monolithic functions.
* If a function becomes too long (~40 lines), split it.
* Avoid deeply nested logic; extract helper functions.

---

## Type & Style Rules

* Use Python 3.11+ syntax.
* Use `X | None` instead of `Optional[X]`.
* Prefer explicit variable names over short unclear names.
* Avoid unnecessary inline comments.
* Do not duplicate docstrings in comments.

---

## Path & Config Rules

* Do not hardcode absolute file paths.
* Do not hardcode dataset or artifact locations.
* Use environment variables or config values.
* Assume datasets and artifacts may exist outside the repository.
* Fail clearly if required files are missing.

---

## Testing Rules

* Test logic at the function level where possible.
* Cover edge cases:

  * empty input
  * OOV-only input
  * invalid input
  * missing artifact files
* Prefer small focused tests over large end-to-end-only tests.
* If a function is hard to test, simplify or split it.

---

## Refactoring Rules

* Preserve existing behavior before modifying logic.
* Refactor incrementally, not all at once.
* Reuse notebook logic instead of rewriting from scratch.
* Extract clean functions from notebook cells.
* Do not hide known limitations.
* Clearly document model limitations in README.

---

## File Responsibility Rules

* Each file must have a single responsibility.
* Do not mix training, inference, and evaluation in one file.

Recommended structure:

* preprocessing.py -> text cleaning, tokenization
* embedding.py -> Word2Vec loading, vectorization
* model.py -> PyTorch model definitions
* predictor.py -> end-to-end inference pipeline
* train_toxicity_model.py -> training script
* predict_comment.py -> CLI inference

---

## Implementation Rules

* Keep changes minimal and local.
* Prefer composable pipelines over complex class hierarchies.
* Make input/output boundaries explicit.
* Optimize for readability over cleverness.
* Keep the code understandable for a junior backend developer.

---

## Do Not (Critical)

* Do not add new features outside the current phase.
* Do not introduce FastAPI, RAG, or vector DB yet.
* Do not replace the existing model.
* Do not introduce new ML models (BERT, transformers, etc.).
* Do not over-engineer architecture.
* Do not create unnecessary abstraction layers.
* Do not restructure the entire repository at once.
* Do not modify multiple unrelated parts in one step.

---

## Anti-Patterns to Avoid

* Rewriting working notebook logic from scratch
* Mixing multiple responsibilities in a single file
* Hardcoding file paths or parameters
* Adding unused or placeholder architecture
* Premature optimization
* Making the system "future-ready" before it works

---

## Validation

* Inference must run end-to-end on a single comment.
* Output must follow `{label, score}` format.
* Pipeline must not crash on edge cases.
* Model loading must fail clearly if artifacts are missing.

---

## Commit Rules

* Use prefixes: feat, refactor, fix, chore, style
* Write commit messages in Korean
* Keep commits small and focused (one change per commit)

---

## Summary

Build a working, minimal inference pipeline first.

Do NOT expand scope.

Stability and clarity are more important than completeness.
