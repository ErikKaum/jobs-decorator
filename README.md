# jobs-decorator

Turn Python functions into remote [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) with a single decorator.

```python
from jobs_decorator import job

@job
def add(a: int, b: int) -> int:
    return a + b

# Run locally
add(2, 3)  # 5

# Run on HF infrastructure
handle = add.remote(2, 3)
result = handle.result()  # 5
```

## Installation

```bash
pip install jobs-decorator
```

Requires a [Hugging Face Pro](https://huggingface.co/pro) or Enterprise account and a valid `HF_TOKEN` (via `huggingface-cli login`).

## Usage

### Basic

The `@job` decorator gives any function `.remote()` and `.local()` methods. Calling the function directly runs it locally.

```python
from jobs_decorator import job

@job(flavor="a10g-small", timeout="2h", dependencies=["torch"])
def train(model_name: str, lr: float = 5e-5):
    import torch
    # ... training code ...
    return {"loss": 0.01}

handle = train.remote("bert-base-uncased", lr=3e-5)
print(handle.url)       # link to the job on HF
print(handle.status())  # RUNNING, COMPLETED, ERROR, etc.

for line in handle.logs():
    print(line)

result = handle.result()  # blocks until done, returns the function's return value
```

### Parameters

| Parameter | Description |
|---|---|
| `flavor` | Hardware flavor (`"cpu-basic"`, `"a10g-small"`, `"l4x4"`, etc.). Default `"cpu-basic"`. |
| `timeout` | Max runtime (`"30m"`, `"2h"`, `"1d"`). Default `"30m"`. |
| `dependencies` | Explicit pip dependencies. If omitted, reads from `pyproject.toml`. |
| `python` | Python version constraint. Defaults to pinning the local version for cloudpickle compatibility. |
| `extras` | Optional dependency groups from `pyproject.toml` to include. |
| `dependency_group` | PEP 735 dependency group to include. |
| `locked` | If `True`, pin all versions from `uv.lock`. |
| `image` | Docker image. Switches to `run_job()` backend. |
| `sidecar` | A `Sidecar` service to run alongside the function (see below). |
| `volumes` | List of `huggingface_hub.Volume` to mount. |
| `env` | Environment variables for the job. |
| `secrets` | Secret environment variables (encrypted). |
| `namespace` | HF namespace (user or org) for the job. |

### Dependency resolution

By default, dependencies are read from your project's `pyproject.toml`. You can override this:

```python
# Explicit deps
@job(dependencies=["torch", "transformers"])
def train(): ...

# From pyproject.toml extras
@job(extras=["gpu"])
def train(): ...

# Pin exact versions from uv.lock
@job(locked=True)
def train(): ...
```

### Sidecar services

For inference servers like TEI or TGI that expose an HTTP API but have no Python package, use `Sidecar`:

```python
from jobs_decorator import job, Sidecar

tei = Sidecar(
    image="ghcr.io/huggingface/text-embeddings-inference:1.5",
    command="text-embeddings-router --model-id BAAI/bge-large-en-v1.5 --port 8080",
    port=8080,
    readiness_endpoint="/health",
)

@job(sidecar=tei, flavor="a10g-small", dependencies=["requests"])
def embed(texts: list[str]) -> list[list[float]]:
    import requests
    resp = requests.post("http://localhost:8080/embed", json={"inputs": texts})
    return resp.json()
```

HF Jobs only supports a single container, so the sidecar runs as a background process within that container. The library starts the service, polls the readiness endpoint until healthy, then runs your function.

All `Sidecar` fields are required -- no magic defaults.

### Volumes

Mount HF datasets, models, or storage buckets into the job:

```python
from huggingface_hub import Volume
from jobs_decorator import job

@job(
    flavor="cpu-basic",
    volumes=[Volume(type="dataset", source="HuggingFaceFW/fineweb", mount_path="/data")],
)
def process():
    import os
    return os.listdir("/data")
```

### JobHandle

`.remote()` returns a `JobHandle` with these methods:

| Method | Description |
|---|---|
| `handle.status()` | Current stage: `RUNNING`, `COMPLETED`, `ERROR`, `CANCELED` |
| `handle.logs()` | Iterator over log lines |
| `handle.metrics()` | Resource usage (CPU, memory, GPU) |
| `handle.wait()` | Block until terminal state |
| `handle.result()` | Block and return deserialized return value |
| `handle.cancel()` | Cancel the job |

## How it works

1. Your function and arguments are serialized with `cloudpickle` and embedded into a PEP 723 UV script.
2. The script is submitted to HF Jobs via `run_uv_job()` (or `run_job()` for Docker/sidecar).
3. On the remote, the function is deserialized, executed, and the return value is uploaded to a private HF dataset repo (`{namespace}/jobs-decorator-results`).
4. `handle.result()` downloads and deserializes the return value.

The remote Python version is pinned to match your local version to ensure cloudpickle compatibility.

## Examples

See the [`examples/`](examples/) directory:

- [`basic.py`](examples/basic.py) -- minimal usage
- [`gpu_training.py`](examples/gpu_training.py) -- GPU training with torch/transformers
- [`parallel_jobs.py`](examples/parallel_jobs.py) -- submit multiple jobs concurrently
- [`with_secrets.py`](examples/with_secrets.py) -- environment variables and secrets
- [`pyproject_deps.py`](examples/pyproject_deps.py) -- automatic dependency resolution
- [`tei_embeddings.py`](examples/tei_embeddings.py) -- sidecar pattern with TEI
- [`volumes.py`](examples/volumes.py) -- mounting HF datasets
