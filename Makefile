PY ?= .venv/bin/python

.PHONY: verify test
# fast, deterministic, offline: the gate for every change
verify:
	$(PY) -m compileall -q opendubstage scripts tests
	$(PY) -m pytest -q

test: verify
