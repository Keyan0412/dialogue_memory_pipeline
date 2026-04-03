# Dialogue Memory Pipeline

`dialogue_memory_pipeline` turns a dialogue transcript into:

- candidate topic-shift boundaries
- per-utterance local discourse states
- finalized dialogue segments
- episode-style memory records for each segment

The package is built around an OpenAI-compatible JSON LLM client and is intended for dialogue understanding, segmentation, and memory construction workflows.

## What It Does

Given a sequence of utterances, the pipeline runs four stages:

1. Candidate boundary generation
   Scores every possible boundary between adjacent utterances and keeps only the highest-confidence candidates.
2. Local state extraction
   Extracts a structured state for each utterance, including topic, intent, entities, cue markers, and obligation signals.
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

## Installation

Install the package from source:

```bash
git clone https://github.com/Keyan0412/dialogue_memory_pipeline.git
cd dialogue_memory_pipeline
python -m venv .venv
source .venv/bin/activate
pip install .
```

Required dependencies:

- `openai`
- `python-dotenv`

## Environment Variables

The pipeline uses an OpenAI-compatible API.

Supported environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `OPENAI_MODEL` (optional when using `from_env`)

Example `.env`:

```env
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen3.5-plus
```

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
)

pipeline = DialogueSegmentationPipeline.from_env(config=config)
result = pipeline.run(dialogue)

print(result["segments"])
print(result["episodes"])
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

result = pipeline.run(dialogue)
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

- `run(utterances) -> dict`

### `PipelineConfig`

Current configuration fields:

```python
PipelineConfig(
    top_p_candidates=0.30,
    min_candidate_score=0.20,
    right_preview_window=3,
    min_segment_len=2,
)
```

Field meanings:

- `top_p_candidates`: Fraction of available boundaries to keep after scoring. If a dialogue has `N` utterances, there are `N - 1` possible boundaries. The retained candidate count is `ceil((N - 1) * top_p_candidates)`, with a minimum of 1 whenever any boundary exists.
- `min_candidate_score`: Minimum boundary score to keep before the top-p cap is applied.
- `right_preview_window`: Number of right-side local states shown to the transition judge when evaluating a candidate split.
- `min_segment_len`: Minimum allowed segment length used during segmentation and cleanup merge.

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
- `salient_entities`
- `cue_markers`
- `obligation.opens`
- `obligation.resolves`

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

- `episode_id`
- `utterance_span`
- `utterances`
- `retrieval_summary`
- `key_entities`
- `importance`

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
  Packaged sample dialogue data included in the wheel
- `examples/`
  Small usage examples
- `scripts/`
  Runnable scripts for demos and module-level testing
- `tests/`
  Local test coverage for packaging and defensive parsing behavior
- `outputs/`
  Generated artifacts

## Model and Provider Notes

- The package expects an OpenAI-compatible API that supports JSON responses.
- `OPENAI_BASE_URL` lets you point the client at compatible providers.
- `from_env()` defaults to `OPENAI_MODEL` when set, otherwise it falls back to `qwen3.5-plus`.

## Current Limitations

- The implementation is fully LLM-driven; there is no local fallback model path in the package.
- Transition-judge behavior is model-dependent because split decisions are generated by the LLM.
- The implementation depends on an OpenAI-compatible JSON-capable model endpoint.

## License

This project is released under the Apache License 2.0. See [`LICENSE`](LICENSE).
