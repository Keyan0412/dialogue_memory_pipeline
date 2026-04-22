# LongMemEval Experiment Plan

## Goal

Use `LongMemEval` as the next-stage benchmark to test whether ChronoMem's upstream memory-construction pipeline improves memory recall on long conversational histories.

The immediate target is:

- build a retrieval-ready memory index from LongMemEval histories
- run session-level recall experiments first
- analyze whether `episode`-level memory construction is better than raw-session retrieval

This stage focuses on retrieval, not full question answering.

## Why LongMemEval

LongMemEval is a benchmark for long-term conversational memory with five major ability types:

- information extraction
- multi-session reasoning
- temporal reasoning
- knowledge updates
- abstention

Its released data provides:

- `question`
- `question_type`
- `haystack_session_ids`
- `haystack_dates`
- `haystack_sessions`
- `answer_session_ids`

This makes it suitable for evaluating the indexing and retrieval stages separately from the reading stage.

## Core Experimental Principle

ChronoMem should treat each LongMemEval history session as one input dialogue to the current upstream pipeline.

That means:

- one `haystack_session` = one `dialogue`
- one LongMemEval `session_id` = one ChronoMem `dialogue_id`
- one session is transformed into one or more `episodes`

The retrieval index is then built over `episodes`, while evaluation first happens at the `session` level using LongMemEval's official `answer_session_ids`.

## Stage 1: Dataset Adapter

Create a LongMemEval adapter that converts the original benchmark format into ChronoMem-ready pipeline inputs.

For each LongMemEval instance:

- read `question_id`
- read `question`
- read `question_type`
- read `haystack_session_ids`
- read `haystack_dates`
- read `haystack_sessions`
- read `answer_session_ids`

For each session inside `haystack_sessions`:

- map turn index to `turn_id`
- map turn `role` to ChronoMem `speaker`
- map turn `content` to ChronoMem `text`
- set `dialogue_id = session_id`

Recommended normalized intermediate schema:

```json
{
  "question_id": "string",
  "question": "string",
  "question_type": "string",
  "question_date": "string",
  "answer_session_ids": ["session_id_1", "session_id_2"],
  "sessions": [
    {
      "dialogue_id": "session_id_1",
      "date": "timestamp",
      "turns": [
        {
          "turn_id": 0,
          "speaker": "user",
          "text": "..."
        }
      ]
    }
  ]
}
```

## Stage 2: Memory Construction

Run the current pipeline on every unique session.

For each session:

1. call `pipeline.run(turns, dialogue_id=session_id)`
2. export normalized episode records
3. attach the original session timestamp if needed

Expected output per episode:

- `dialogue_id`
- `episode_id`
- `turn_start`
- `turn_end`
- `relative_start`
- `relative_end`
- `stable_topic`
- `discourse_goal`
- `retrieval_summary_en`
- `key_entities_en`
- `importance`
- `token_estimate`

Recommended output artifact:

- `outputs/longmemeval/episodes.jsonl`

Each record should also preserve:

- `source_session_id`
- `source_session_date`

In practice, `source_session_id` can be the same as `dialogue_id`.

## Stage 3: Retrieval Index Construction

Build a retrieval index over `episodes`, not directly over sessions.

### Initial retrieval fields

Use these as the first retrieval key:

- `retrieval_summary_en`

Use these as auxiliary fields:

- `key_entities_en`
- `stable_topic`
- `discourse_goal`

### Initial retrievers

Implement the following in order:

1. BM25 baseline
2. dense retriever baseline
3. hybrid retriever baseline

The first goal is not SOTA. The first goal is to produce a stable benchmark pipeline.

## Stage 4: Retrieval and Aggregation

For each LongMemEval question:

1. use `question` as the retrieval query
2. retrieve top-k `episodes`
3. aggregate episode scores back to `session_id`
4. rank sessions
5. compare ranked sessions against `answer_session_ids`

Recommended first aggregation strategies:

- max score per session
- sum of top-n episode scores per session

Start with `max`.

Recommended first `k` values:

- `k = 1`
- `k = 5`
- `k = 10`
- `k = 20`

## Stage 5: Metrics

### Primary metrics

For the first round, report:

- `Recall@1`
- `Recall@5`
- `Recall@10`
- `Recall@20`
- `Hit@1`
- `Hit@5`
- `Hit@10`
- `Hit@20`

### Breakdown metrics

Report the same metrics by `question_type`:

- `single-session-user`
- `single-session-assistant`
- `single-session-preference`
- `temporal-reasoning`
- `knowledge-update`
- `multi-session`

Abstention questions should be excluded from retrieval recall unless a separate negative-retrieval evaluation is intentionally designed.

## Stage 6: Baselines

The most important comparison is not only retriever A vs retriever B.

It is:

### Baseline A: Raw session retrieval

Use each original LongMemEval session as one retrieval unit.

Possible key:

- concatenated session text
- session summary if you later build one

### Baseline B: ChronoMem episode retrieval

Use ChronoMem-produced `episodes` as retrieval units and then aggregate back to sessions.

This comparison answers the real research question:

Does memory construction improve recall over naive retrieval on raw histories?

## Stage 7: Ablations

After the first stable benchmark works, run ablations in this order.

### Ablation 1: Retrieval key composition

Compare:

- summary only
- summary + entities
- summary + entities + topic + discourse goal

### Ablation 2: Granularity

Compare:

- session-level index
- episode-level index

### Ablation 3: Pipeline segmentation effect

Compare:

- one whole session as one memory
- ChronoMem segmented episodes

This isolates whether segmentation itself provides value.

### Ablation 4: Importance weighting

Test whether `importance` should affect retrieval score or reranking score.

### Ablation 5: Temporal features

Later, when time metadata is strengthened, compare:

- no temporal prior
- relative-position prior
- timestamp-aware prior

## Stage 8: Error Analysis

For failed retrieval examples, inspect:

- whether the correct session was never converted into a useful episode
- whether segmentation split the evidence too aggressively
- whether the summary omitted the answer-bearing fact
- whether retrieval missed the right episode despite a good summary
- whether aggregation from episode to session failed
- whether the question requires temporal reasoning not yet captured by the current metadata

Save failure reports with:

- `question_id`
- `question_type`
- query text
- gold sessions
- retrieved sessions
- top retrieved episodes
- top episode summaries

## Recommended File Outputs

Suggested artifacts for the experiment:

- `outputs/longmemeval/adapted_instances.jsonl`
- `outputs/longmemeval/episodes.jsonl`
- `outputs/longmemeval/retrieval_results.jsonl`
- `outputs/longmemeval/metrics.json`
- `outputs/longmemeval/error_analysis.jsonl`

## Minimum Viable Experiment

The first complete experiment should be:

1. load `longmemeval_s.json`
2. normalize all sessions into ChronoMem dialogue format
3. run pipeline on unique sessions
4. export `episodes.jsonl`
5. build a BM25 index over `retrieval_summary_en`
6. retrieve top-k episodes with `question`
7. aggregate retrieved episodes to session ids
8. compute session-level `Recall@k` and `Hit@k`
9. report metrics by `question_type`

If this works, the benchmark loop is already good enough to support later dense, hybrid, temporal, and reranking experiments.

## Immediate Implementation Tasks

1. Write a LongMemEval adapter script.
2. Write a batch memory-construction script for unique sessions.
3. Write a BM25 retrieval script over exported episodes.
4. Write a session-level evaluation script against `answer_session_ids`.
5. Add per-type metrics and failure-case dumps.

## Success Criteria

This stage is successful when:

- LongMemEval can be ingested without manual relabeling
- ChronoMem exports a stable `episodes.jsonl`
- retrieval runs reproducibly on the same benchmark split
- metrics can be reproduced from saved logs
- raw-session vs episode-level retrieval can be compared directly
