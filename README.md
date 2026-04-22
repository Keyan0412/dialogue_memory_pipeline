# ChronoMem

ChronoMem is a monorepo for long-dialogue memory construction, storage, retrieval, and evaluation work.

## Layout

- `src/dialogue_memory_pipeline/`
  Upstream memory-construction package code.
- `tests/dialogue_memory_pipeline/`
  Package tests.
- `scripts/dialogue_memory_pipeline/`
  Runnable entrypoints for demos and data export.
- `examples/dialogue_memory_pipeline/`
  Minimal usage examples.
- `docs/`
  Project and package documentation.
- `outputs/`
  Generated local artifacts.

## Current Packages

### `dialogue_memory_pipeline`

The upstream package turns a long dialogue into:

- candidate boundaries
- local discourse states
- finalized segments
- normalized episodic memory records

Its import path remains:

```python
from dialogue_memory_pipeline import DialogueSegmentationPipeline
```
