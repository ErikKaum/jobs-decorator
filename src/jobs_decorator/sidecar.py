"""Sidecar — run a service container alongside your function."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sidecar:
    """A service that runs alongside the user's function inside the same container.

    Since HF Jobs only supports a single container per job, the sidecar process
    is started in the background within that container before the user's function
    executes. The library waits for the service to pass a readiness check before
    running the function.

    This is useful for inference servers (TEI, TGI, etc.) that expose an HTTP
    API but have no Python package — your function talks to them over localhost.

    Parameters
    ----------
    image : str
        Docker image for the sidecar service (e.g.
        ``"ghcr.io/huggingface/text-embeddings-inference:1.5"``).
        This becomes the container image for the entire job.
    command : str
        Shell command to start the service (e.g.
        ``"text-embeddings-router --model-id BAAI/bge-large-en-v1.5 --port 8080"``).
        Runs in the background before the user's function.
    port : int
        Port the service listens on. Used to build the readiness probe URL
        (``http://localhost:{port}{readiness_endpoint}``).
    readiness_endpoint : str
        HTTP path polled until the service is ready (e.g. ``"/health"``).
        The library retries every 2 seconds until it gets a successful response.
    """

    image: str
    command: str
    port: int
    readiness_endpoint: str
