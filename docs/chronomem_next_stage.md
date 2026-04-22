# ChronoMem Next Stage

## Project Name

The unified memory system built on top of `dialogue_memory_pipeline` will use the project name:

`ChronoMem`

Short description:

`ChronoMem` is a time-aware, budget-aware memory system for very long dialogues. It treats the current repository as the upstream memory-construction layer and extends it into a full retrieval and reranking stack.

## Positioning

The current repository should remain the upstream producer of structured episodic memories.

`ChronoMem` expands the system into three connected layers:

1. Memory construction
2. Memory indexing and storage
3. Memory retrieval and reranking

This keeps the current pipeline focused while making room for long-context retrieval experiments, evaluation, and serving logic.

## Component Map

### 1. Upstream: Memory Construction

Repository role today:

- dialogue segmentation
- local state extraction
- transition judgment
- episodic memory building

Recommended logical package name:

- `chronomem.pipeline`

Main outputs:

- `segments`
- `episodes`
- compact retrieval summaries
- bilingual key entities
- segment-level discourse state

### 2. Midstream: Memory Store and Indexing

Recommended logical package name:

- `chronomem.store`

Responsibilities:

- normalize episode records
- add stable `dialogue_id` and `episode_id`
- add temporal metadata
- estimate token cost
- build sparse, dense, and hybrid indexes
- persist memory artifacts in JSONL / Parquet / SQLite / vector store

Core record fields:

- `dialogue_id`
- `episode_id`
- `turn_start`
- `turn_end`
- `timestamp_start`
- `timestamp_end`
- `retrieval_summary_zh`
- `retrieval_summary_en`
- `key_entities_zh`
- `key_entities_en`
- `importance`
- `stable_topic`
- `discourse_goal`
- `open_obligations`
- `token_estimate`
- `last_access_time`
- `access_count`

### 3. Downstream: Retriever

Recommended logical package name:

- `chronomem.retrieve`

Responsibilities:

- query normalization
- sparse retrieval
- dense retrieval
- hybrid candidate generation
- temporal priors
- session-aware recall

Candidate retriever modules:

- `bm25_retriever`
- `dense_retriever`
- `hybrid_retriever`
- `temporal_retriever`

### 4. Downstream: Reranker

Recommended logical package name:

- `chronomem.rerank`

Responsibilities:

- cross-encoder reranking
- temporal reranking
- budget-aware candidate selection
- diversity-aware final set construction

Candidate reranker modules:

- `cross_encoder_reranker`
- `temporal_reranker`
- `budget_aware_reranker`
- `coverage_reranker`

### 5. Evaluation

Recommended logical package name:

- `chronomem.eval`

Responsibilities:

- retrieval benchmark runs
- rerank ablations
- budget-aware evaluation
- temporal sensitivity analysis
- per-query reports

Key metrics:

- recall@k
- main recall@k
- hit@k
- MRR
- nDCG
- token budget utilization
- relevance-per-token
- recency sensitivity

## Recommended Repository Evolution

### Option A: Keep This Repository as the Upstream Core

Recommended when:

- segmentation and memory construction are still changing quickly
- retrieval experiments need to iterate independently

Suggested structure:

```text
dialogue_memory_pipeline/
  docs/
    chronomem_next_stage.md
  src/dialogue_memory_pipeline/
  scripts/
  outputs/
  exports/
    episodes.jsonl
```

And then create a separate retrieval repository:

```text
chronomem/
  src/chronomem/
    store/
    retrieve/
    rerank/
    eval/
  configs/
  experiments/
  reports/
```

### Option B: Evolve into a Monorepo Later

Recommended when:

- the episode schema becomes stable
- retrieval and reranking code needs tight integration with upstream outputs

Suggested structure:

```text
chronomem/
  packages/
    dialogue_memory_pipeline/
    chronomem_store/
    chronomem_retrieve/
    chronomem_rerank/
    chronomem_eval/
  docs/
  configs/
  experiments/
```

For the next stage, Option A is the safer choice.

## Next Stage Milestones

### Milestone 1: Stabilize Upstream Exports

Add a stable episode export format for downstream retrieval:

- `exports/episodes.jsonl`
- one normalized record per memory episode
- include dialogue id, episode id, turn span, summaries, entities, importance, and temporal placeholders

### Milestone 2: Add Temporal Metadata

Extend episode records with:

- absolute timestamps when available
- turn-based relative time
- session distance
- ingest time

### Milestone 3: Build First Retrieval Baseline

Implement:

- sparse retrieval baseline
- dense retrieval baseline
- hybrid top-k candidate generation

### Milestone 4: Add Budget-Aware Reranking

Implement a reranker that scores candidates using:

- semantic relevance
- temporal relevance
- importance
- token cost

Target objective:

- maximize retrieval value under a fixed context budget

### Milestone 5: Add Evaluation Harness

Create a reproducible evaluation pipeline for:

- long-dialogue memory retrieval
- temporal retrieval behavior
- budget-aware reranking quality

## Naming Guidance for Subsystems

Recommended subsystem names:

- `ChronoMem Pipeline`
- `ChronoMem Store`
- `ChronoMem Retrieve`
- `ChronoMem Rerank`
- `ChronoMem Eval`

Recommended repo naming:

- current upstream repo: keep as `dialogue_memory_pipeline`
- future unified repo: `chronomem`

## Immediate Action Items

The next concrete implementation steps for this repository are:

1. Add a normalized episode export script.
2. Add `dialogue_id` and `episode_id` stability guarantees.
3. Add temporal placeholder fields to episode outputs.
4. Keep retrieval and reranking logic out of the upstream package for now.

## Summary

`ChronoMem` should be treated as the unified system name, while the current repository remains the upstream memory-construction component.

The core ChronoMem components are:

1. `ChronoMem Pipeline`
2. `ChronoMem Store`
3. `ChronoMem Retrieve`
4. `ChronoMem Rerank`
5. `ChronoMem Eval`
