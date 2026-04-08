# CLAUDE.md

## Project overview

`jobs-decorator` is a Python library that turns functions into remote Hugging Face Jobs via a `@job` decorator. Functions gain `.remote()` and `.local()` methods.

## Architecture

- `src/jobs_decorator/decorator.py` -- `@job` decorator and `JobFunction` class
- `src/jobs_decorator/backend.py` -- dispatches to `run_uv_job()` (default), `run_job()` (Docker), or sidecar path
- `src/jobs_decorator/handle.py` -- `JobHandle` for tracking/retrieving results
- `src/jobs_decorator/sidecar.py` -- `Sidecar` dataclass for service containers
- `src/jobs_decorator/script.py` -- generates PEP 723 UV scripts with inline deps
- `src/jobs_decorator/dependencies.py` -- resolves deps from pyproject.toml / uv.lock

Three submission paths:
1. **UV (default)**: generates a PEP 723 script, submits via `run_uv_job()`
2. **Docker**: uploads script to HF, submits via `run_job()` with a shell command
3. **Sidecar**: starts service in background, installs uv, runs PEP 723 script via `uv run`

Results flow through a private HF dataset repo (`{namespace}/jobs-decorator-results`).

## Key design decisions

- Python version is pinned to the local major.minor by default (e.g. `==3.10`) because cloudpickle bytecode is not compatible across Python versions
- `HF_TOKEN` is injected via `huggingface_hub.get_token()` (not `api.token` which returns None in some auth configurations)
- Sidecar uses `generate_uv_script()` (same as the UV path) to avoid shell-quoting issues with PEP 508 dependency markers
- Poll interval for `handle.wait()` / `handle.result()` is 15 seconds to avoid hitting the `/whoami` rate limit

## Commands

```bash
uv run python examples/basic.py       # run an example
uv run python -c "from jobs_decorator import job"  # verify imports
```

## Dependencies

Defined in `pyproject.toml`: `huggingface_hub>=0.30.0`, `cloudpickle>=3.0`, `tomli>=2.0` (Python <3.11).
