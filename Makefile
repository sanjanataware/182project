PY ?= python

# Every table and figure in the report, from the committed per-prompt scores (laptop, no GPU).
figures:
	$(PY) ihc/reproduce.py

# Same, plus the supplementary analyze.py tables.
all:
	$(PY) ihc/reproduce.py --all

# Public datasets -> data/ (checked against data/SHA256SUMS).
data:
	bash scripts/fetch_data.sh

# Raw responses (GitHub Release) -> re-run the IHEval checkers -> scored/, then the figures.
score: data
	bash scripts/fetch_responses.sh
	$(PY) ihc/score.py responses/gpt-oss-20b__*.jsonl
	$(PY) ihc/reproduce.py

test: data
	$(PY) ihc/tests/test_pipeline.py

.PHONY: figures all data score test
