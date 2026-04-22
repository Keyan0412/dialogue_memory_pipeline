# Dialogue Memory Pipeline

`dialogue_memory_pipeline` turns a dialogue transcript into:

- candidate topic-shift boundaries
- per-utterance local discourse states
- finalized dialogue segments
- episode-style memory records for each segment

The package is built around an OpenAI-compatible JSON LLM client and is intended for dialogue understanding, segmentation, and memory construction workflows.

## Project Status

This package is currently in active development and should be treated as an alpha release.

- APIs, behavior, and output formats may still change.
- The package is not yet production-ready.
- At the moment, the only provider setup that is tested and supported is Bailian (DashScope) using its OpenAI-compatible endpoint.

If you publish this package, it is best to assume early-adopter usage rather than stable general availability.

## What It Does

Given a sequence of utterances, the pipeline runs four stages:

1. Candidate boundary generation
   Scores every possible boundary between adjacent utterances and keeps only the highest-confidence candidates.
2. Local state extraction
   Extracts a compact structured state for each utterance, centered on topic, intent, and obligation signals.
3. Transition judgment and segmentation
   Walks candidate boundaries in order and decides whether each one starts a new segment.
4. Episodic memory building
   Produces one memory record per final segment.

The top-level entrypoint is `DialogueSegmentationPipeline`.

## Features

- End-to-end dialogue segmentation and memory generation
- OpenAI-compatible client with optional custom `base_url`
- Structured JSON outputs for every pipeline stage
- Configurable candidate selection thresholding
- Simple API for loading dialogues from JSON files

## Monorepo Placement

Within ChronoMem, this is now a subpackage rather than a standalone project.

- package code: `src/dialogue_memory_pipeline/`
- tests: `tests/dialogue_memory_pipeline/`
- runnable scripts: `scripts/dialogue_memory_pipeline/`
- examples: `examples/dialogue_memory_pipeline/`

## Installation

From the ChronoMem repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai python-dotenv pytest
export PYTHONPATH=src
```

Required dependencies:

- `openai`
- `python-dotenv`

## Environment Variables

The pipeline currently expects a Bailian (DashScope) API key and endpoint through an OpenAI-compatible interface.

Supported environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (default fallback for all stages)
- `OPENAI_MODEL_CANDIDATE` (optional stage override)
- `OPENAI_MODEL_LOCAL_STATE` (optional stage override)
- `OPENAI_MODEL_TRANSITION` (optional stage override)
- `OPENAI_MODEL_MEMORY` (optional stage override)

Example `.env`:

```env
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen3.5-plus
OPENAI_MODEL_CANDIDATE=qwen3.5-plus
OPENAI_MODEL_LOCAL_STATE=qwen3.5-flash
OPENAI_MODEL_TRANSITION=qwen3.5-plus
OPENAI_MODEL_MEMORY=qwen3.5-plus
```

Stage model selection in `from_env()` uses this fallback order:

- stage-specific env var, if set
- `OPENAI_MODEL`
- built-in default `qwen3.5-plus`

At this stage, other OpenAI-compatible providers may or may not work, but they are not yet officially supported by this package.

## Quick Start

### Use the pipeline with environment variables

```python
from dialogue_memory_pipeline import (
    DialogueSegmentationPipeline,
    PipelineConfig,
    load_sample_dialogue,
)

dialogue = load_sample_dialogue()

config = PipelineConfig(
    top_p_candidates=0.30,
    min_candidate_score=0.20,
    right_preview_window=3,
    min_segment_len=2,
    local_state_chunk_size=8,
    local_state_max_parallel=4,
)

pipeline = DialogueSegmentationPipeline.from_env(config=config)
result = pipeline.run(dialogue, dialogue_id="dlg_sample")
memories = pipeline.extract_memories(result, language="en")

print(result["segments"])
print(memories)
```

### Use the pipeline with explicit credentials

```python
from dialogue_memory_pipeline import DialogueSegmentationPipeline, load_sample_dialogue

dialogue = load_sample_dialogue()

pipeline = DialogueSegmentationPipeline.from_openai(
    model="qwen3.5-plus",
    api_key="YOUR_API_KEY",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

result = pipeline.run(dialogue, dialogue_id="dlg_sample")
```

## Public API

The package exports:

```python
from dialogue_memory_pipeline import (
    DialogueSegmentationPipeline,
    PipelineConfig,
    load_dialogue,
    load_sample_dialogue,
)
```

### `DialogueSegmentationPipeline`

Constructors:

- `DialogueSegmentationPipeline(llm, config=None)`
- `DialogueSegmentationPipeline.from_env(model=None, config=None)`
- `DialogueSegmentationPipeline.from_openai(model, api_key=None, base_url=None, config=None)`

Main method:

- `run(utterances, dialogue_id=None) -> dict`
- `extract_memories(result, language=None) -> list`
- `normalize_episode_records(result, dialogue_id=None) -> list`
- `export_episodes(result, path, dialogue_id=None) -> int`

### `PipelineConfig`

Current configuration fields:

```python
PipelineConfig(
    top_p_candidates=0.30,
    min_candidate_score=0.20,
    right_preview_window=3,
    min_segment_len=2,
    local_state_chunk_size=0,
    local_state_max_parallel=1,
    local_state_transport="default",
)
```

Field meanings:

- `top_p_candidates`: Fraction of available boundaries to keep after scoring. If a dialogue has `N` utterances, there are `N - 1` possible boundaries. The retained candidate count is `ceil((N - 1) * top_p_candidates)`, with a minimum of 1 whenever any boundary exists.
- `min_candidate_score`: Minimum boundary score to keep before the top-p cap is applied.
- `right_preview_window`: Number of right-side local states shown to the transition judge when evaluating a candidate split.
- `min_segment_len`: Minimum allowed segment length used during segmentation and cleanup merge.
- `local_state_chunk_size`: Chunk size for local state extraction. `0` means send the whole dialogue in one request.
- `local_state_max_parallel`: Maximum parallel chunk requests during local state extraction.
- `local_state_transport`: Transport used by local state extraction. `default` uses the normal OpenAI-compatible endpoint; `bailian_batch_chat` switches local state requests to DashScope Batch Chat.

## Input Format

`load_sample_dialogue()` loads the packaged example dialogue shipped in the wheel.

`load_dialogue(...)` expects a JSON file containing a list of utterances:

```json
[
  {
    "turn_id": 0,
    "speaker": "user",
    "text": "I need to reschedule my flight."
  },
  {
    "turn_id": 1,
    "speaker": "assistant",
    "text": "Sure, what is your booking number?"
  }
]
```

Each item must contain:

- `turn_id`
- `speaker`
- `text`

## Output Structure

`pipeline.run(...)` returns a dictionary with these top-level keys:

- `dialogue_id`
- `candidates`
- `local_states`
- `decisions`
- `segments`
- `episodes`
- `timing`

### `candidates`

Each candidate includes:

- `boundary_after_turn`
- `score`
- `left_turn_id`
- `right_turn_id`
- `left_text`
- `right_text`
- `reasoning`
- `source`

### `local_states`

Each local state includes:

- `turn_id`
- `speaker`
- `summary_topic`
- `intent`
- `obligation.opens`
- `obligation.resolves`

`salient_entities` and `cue_markers` may be omitted or empty in compact extraction mode.

### `segments`

Each finalized segment includes:

- `segment_id`
- `utterance_span`
- `utterances`
- `local_states`
- `segment_state`

`segment_state` contains:

- `stable_topic`
- `discourse_goal`
- `focus_topics`
- `entity_core`
- `open_obligations`
- `dominant_relation`

### `episodes`

Each episodic memory record includes:

- `dialogue_id`
- `episode_id`
- `segment_id`
- `episode_index`
- `episode_count`
- `utterance_span`
- `turn_start`
- `turn_end`
- `utterance_count`
- `relative_start`
- `relative_end`
- `stable_topic`
- `discourse_goal`
- `open_obligations`
- `utterances`
- `retrieval_summary_zh`
- `retrieval_summary_en`
- `key_entities_zh`
- `key_entities_en`
- `importance`
- `token_estimate`

### Normalized episode export

Use `DialogueSegmentationPipeline.export_episodes(...)` to write downstream-friendly JSONL:

```python
result = pipeline.run(dialogue, dialogue_id="dlg_sample")
row_count = pipeline.export_episodes(result, "outputs/episodes.jsonl")
print(row_count)
```

Each JSONL record contains a normalized retrieval-facing schema with stable dialogue/episode ids, turn span metadata, segment state, bilingual summaries, entities, and an approximate `token_estimate`.

## Running the Included Scripts

### Full pipeline demo

```bash
python scripts/demo.py
```

Optional flags:

```bash
python scripts/demo.py --output outputs/demo_output.json --model qwen3.5-plus
```

### Candidate boundary generator test

```bash
python scripts/test_candidate_generator.py --top-p 0.30 --min-score 0.40
```

This writes a JSON report with:

- all scored boundaries
- filtered candidate boundaries
- the effective candidate-generation config

## Repository Layout

- `src/dialogue_memory_pipeline/`
  Importable package
- `src/dialogue_memory_pipeline/clients/`
  LLM client adapters
- `src/dialogue_memory_pipeline/core/`
  Shared dataclasses and schemas
- `src/dialogue_memory_pipeline/modules/`
  Pipeline modules for boundary generation, state extraction, transition judgment, and memory building
- `src/dialogue_memory_pipeline/data/`
  Packaged sample dialogue data
- `examples/dialogue_memory_pipeline/`
  Small usage examples
- `scripts/dialogue_memory_pipeline/`
  Runnable scripts for demos and module-level testing
- `tests/dialogue_memory_pipeline/`
  Local test coverage for defensive parsing and export behavior
- `outputs/`
  Generated local artifacts

## Model and Provider Notes

- The current tested provider is Bailian (DashScope) via its OpenAI-compatible endpoint.
- `OPENAI_BASE_URL` should currently point to the Bailian compatible endpoint unless you are experimenting on your own.
- `from_env()` defaults to `OPENAI_MODEL` when set, otherwise it falls back to `qwen3.5-plus`.
- Support for additional providers is not finalized yet.

## Current Limitations

- The project is still in alpha and may change in breaking ways.
- Only Bailian (DashScope) API credentials and endpoint configuration are currently supported.
- The implementation is fully LLM-driven; there is no local fallback model path in the package.
- Transition-judge behavior is model-dependent because split decisions are generated by the LLM.
- The implementation depends on an OpenAI-compatible JSON-capable model endpoint.

## License

This project is released under the Apache License 2.0. See [`LICENSE`](LICENSE).
