"""Compute embeddings using TEI (Text Embeddings Inference) as a sidecar.

TEI is a high-performance inference server for embedding models. Unlike vLLM
or other frameworks, TEI has no Python package — it's a standalone Rust binary
that exposes an HTTP API.

The Sidecar class lets you run TEI (or any HTTP service) alongside your Python
function inside the same HF Job.

How it works under the hood
---------------------------
HF Jobs only supports a single container per job.  When you use `sidecar=`,
the library:

  1. Uses the sidecar's Docker image as the job's container image.
  2. Generates a wrapper command that starts the sidecar process in the
     background.
  3. Polls the readiness endpoint (http://localhost:{port}{readiness_endpoint})
     until the service is healthy.
  4. Installs uv, then uses it to run your function with deps (PEP 723 script).
  5. Uploads the return value so handle.result() can retrieve it.

Because everything runs in one container, your function talks to the sidecar
over localhost.
"""

from jobs_decorator import Sidecar, job

# ---------------------------------------------------------------------------
# Define the sidecar — every field is required, no magic defaults.
#
#   image:              the Docker image to run
#   command:            how to start the service (runs in the background)
#   port:               the port the service listens on
#   readiness_endpoint: HTTP path polled until the service is ready
# ---------------------------------------------------------------------------
tei = Sidecar(
    image="ghcr.io/huggingface/text-embeddings-inference:1.9.1",
    command="text-embeddings-router --model-id BAAI/bge-large-en-v1.5 --port 8080",
    port=8080,
    readiness_endpoint="/health",
)

@job(sidecar=tei, flavor="a10g-small", timeout="30m", dependencies=["requests"])
def embed(texts: list[str]) -> list[list[float]]:
    """Call the TEI sidecar over HTTP to compute embeddings."""
    import requests

    resp = requests.post(
        "http://localhost:8080/embed",
        json={"inputs": texts},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

handle = embed.remote([
    "The food was absolutely wonderful",
    "I would not recommend this place",
    "Quick service and friendly staff",
])
print(f"Embedding job: {handle.url}")

embeddings = handle.result()
print(f"Got {len(embeddings)} embeddings, dim={len(embeddings[0])}")
