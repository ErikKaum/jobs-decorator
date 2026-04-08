"""HF backend — dispatches to run_uv_job() or run_job()."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import cloudpickle
from huggingface_hub import HfApi

from .handle import JobHandle
from .script import generate_docker_script, generate_uv_script
from .sidecar import Sidecar


def submit_job(
    fn: Any,
    args: tuple,
    kwargs: dict,
    *,
    flavor: str,
    timeout: str,
    dependencies: list[str],
    python_requires: str,
    image: Optional[str] = None,
    dockerfile: Optional[str] = None,
    sidecar: Optional[Sidecar] = None,
    volumes: Optional[list] = None,
    env: Optional[dict[str, str]] = None,
    secrets: Optional[dict[str, str]] = None,
    namespace: Optional[str] = None,
) -> JobHandle:
    """Serialize the function and submit it as an HF job.

    Uses run_uv_job() by default (UV-first). Falls back to run_job() when
    image= or dockerfile= is provided.
    """
    api = HfApi()

    # Determine namespace (user or org)
    if namespace is None:
        namespace = api.whoami()["name"]

    # Create a unique result location
    job_uid = uuid.uuid4().hex[:12]
    result_repo_id = f"{namespace}/jobs-decorator-results"
    result_filename = f"results/{job_uid}.pkl"

    # Ensure the results repo exists
    api.create_repo(
        repo_id=result_repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    # Serialize function and arguments
    pickled_fn = cloudpickle.dumps(fn)
    pickled_args = cloudpickle.dumps((args, kwargs))

    # Inject HF_TOKEN into secrets so the remote can upload results
    from huggingface_hub import get_token

    secrets = dict(secrets or {})
    if "HF_TOKEN" not in secrets:
        token = get_token()
        if token:
            secrets["HF_TOKEN"] = token

    if sidecar is not None:
        job_info = _submit_sidecar_job(
            api=api,
            pickled_fn=pickled_fn,
            pickled_args=pickled_args,
            result_repo_id=result_repo_id,
            result_filename=result_filename,
            flavor=flavor,
            timeout=timeout,
            sidecar=sidecar,
            dependencies=dependencies,
            python_requires=python_requires,
            volumes=volumes,
            env=env,
            secrets=secrets,
            namespace=namespace,
        )
    elif image or dockerfile:
        job_info = _submit_docker_job(
            api=api,
            pickled_fn=pickled_fn,
            pickled_args=pickled_args,
            result_repo_id=result_repo_id,
            result_filename=result_filename,
            flavor=flavor,
            timeout=timeout,
            image=image,
            dockerfile=dockerfile,
            volumes=volumes,
            env=env,
            secrets=secrets,
            namespace=namespace,
        )
    else:
        job_info = _submit_uv_job(
            api=api,
            pickled_fn=pickled_fn,
            pickled_args=pickled_args,
            dependencies=dependencies,
            python_requires=python_requires,
            result_repo_id=result_repo_id,
            result_filename=result_filename,
            flavor=flavor,
            timeout=timeout,
            volumes=volumes,
            env=env,
            secrets=secrets,
            namespace=namespace,
        )

    return JobHandle(
        job_id=job_info.id,
        url=job_info.url,
        result_repo_id=result_repo_id,
        result_filename=result_filename,
        _api=api,
    )


def _submit_uv_job(
    *,
    api: HfApi,
    pickled_fn: bytes,
    pickled_args: bytes,
    dependencies: list[str],
    python_requires: str,
    result_repo_id: str,
    result_filename: str,
    flavor: str,
    timeout: str,
    volumes: Optional[list],
    env: Optional[dict[str, str]],
    secrets: Optional[dict[str, str]],
    namespace: str,
) -> Any:
    """Submit via run_uv_job() — the UV-first path."""
    from huggingface_hub import run_uv_job

    script_content = generate_uv_script(
        pickled_fn=pickled_fn,
        pickled_args=pickled_args,
        dependencies=dependencies,
        python_requires=python_requires,
        result_repo_id=result_repo_id,
        result_filename=result_filename,
    )

    # Write the script to a temp file (run_uv_job expects a file path)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="hfjob_"
    ) as f:
        f.write(script_content)
        script_path = f.name

    kwargs: dict[str, Any] = dict(
        flavor=flavor,
        timeout=timeout,
        env=env or {},
        secrets=secrets or {},
        namespace=namespace,
    )
    if volumes:
        kwargs["volumes"] = volumes

    return run_uv_job(script_path, **kwargs)


def _submit_docker_job(
    *,
    api: HfApi,
    pickled_fn: bytes,
    pickled_args: bytes,
    result_repo_id: str,
    result_filename: str,
    flavor: str,
    timeout: str,
    image: Optional[str],
    dockerfile: Optional[str],
    volumes: Optional[list],
    env: Optional[dict[str, str]],
    secrets: Optional[dict[str, str]],
    namespace: str,
) -> Any:
    """Submit via run_job() — the Docker fallback path."""
    from huggingface_hub import run_job

    if dockerfile:
        raise NotImplementedError(
            "Dockerfile-based jobs are not yet supported. "
            "Use image= with a pre-built Docker image instead."
        )

    script_content = generate_docker_script(
        pickled_fn=pickled_fn,
        pickled_args=pickled_args,
        result_repo_id=result_repo_id,
        result_filename=result_filename,
    )

    # Upload the script to the results repo so the container can fetch it
    script_name = f"scripts/{result_filename.split('/')[-1].replace('.pkl', '.py')}"
    api.upload_file(
        path_or_fileobj=script_content.encode(),
        path_in_repo=script_name,
        repo_id=result_repo_id,
        repo_type="dataset",
    )

    # The container downloads and runs the script
    script_url = f"https://huggingface.co/datasets/{result_repo_id}/resolve/main/{script_name}"

    kwargs: dict[str, Any] = dict(
        image=image or "python:3.12",
        command=[
            "bash", "-c",
            f"pip install cloudpickle huggingface_hub && "
            f"curl -sL -H \"Authorization: Bearer $HF_TOKEN\" '{script_url}' -o /tmp/job.py && "
            f"python /tmp/job.py",
        ],
        flavor=flavor,
        timeout=timeout,
        env=env or {},
        secrets=secrets or {},
        namespace=namespace,
    )
    if volumes:
        kwargs["volumes"] = volumes

    return run_job(**kwargs)


def _submit_sidecar_job(
    *,
    api: HfApi,
    pickled_fn: bytes,
    pickled_args: bytes,
    result_repo_id: str,
    result_filename: str,
    flavor: str,
    timeout: str,
    sidecar: Sidecar,
    dependencies: list[str],
    python_requires: str,
    volumes: Optional[list],
    env: Optional[dict[str, str]],
    secrets: Optional[dict[str, str]],
    namespace: str,
) -> Any:
    """Submit a job that starts a sidecar service, then runs the user's function.

    HF Jobs only supports a single container, so the sidecar and the user's
    Python code share the same container.  The generated command:

    1. Starts the sidecar process in the background.
    2. Polls the readiness endpoint until the service is healthy.
    3. Installs Python dependencies (cloudpickle, huggingface_hub).
    4. Downloads and executes the serialized user function.
    """
    from huggingface_hub import run_job

    # Use the same PEP 723 UV script as the normal path — deps and python
    # version are declared as inline metadata comments, so uv handles
    # markers natively and we avoid all shell-quoting issues.
    script_content = generate_uv_script(
        pickled_fn=pickled_fn,
        pickled_args=pickled_args,
        dependencies=dependencies,
        python_requires=python_requires,
        result_repo_id=result_repo_id,
        result_filename=result_filename,
    )

    # Upload the script to the results repo so the container can fetch it
    script_name = f"scripts/{result_filename.split('/')[-1].replace('.pkl', '.py')}"
    api.upload_file(
        path_or_fileobj=script_content.encode(),
        path_in_repo=script_name,
        repo_id=result_repo_id,
        repo_type="dataset",
    )

    script_url = f"https://huggingface.co/datasets/{result_repo_id}/resolve/main/{script_name}"
    readiness_url = f"http://localhost:{sidecar.port}{sidecar.readiness_endpoint}"

    bash_script = (
        f"# Start the sidecar service in the background\n"
        f"{sidecar.command} &\n"
        f"SIDECAR_PID=$!\n"
        f"\n"
        f"# Wait for the service to be ready\n"
        f"echo 'Waiting for sidecar on {readiness_url} ...'\n"
        f"until curl -sf {readiness_url} > /dev/null 2>&1; do\n"
        f"  if ! kill -0 $SIDECAR_PID 2>/dev/null; then\n"
        f"    echo 'Sidecar process exited unexpectedly'; exit 1\n"
        f"  fi\n"
        f"  sleep 2\n"
        f"done\n"
        f"echo 'Sidecar ready'\n"
        f"\n"
        f"# Install uv, then use it to run the PEP 723 script (deps are inline)\n"
        f"curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        f"export PATH=\"$HOME/.local/bin:$PATH\"\n"
        f"curl -sL -H \"Authorization: Bearer $HF_TOKEN\" '{script_url}' -o /tmp/job.py && "
        f"uv run /tmp/job.py\n"
    )

    kwargs: dict[str, Any] = dict(
        image=sidecar.image,
        command=["bash", "-c", bash_script],
        flavor=flavor,
        timeout=timeout,
        env=env or {},
        secrets=secrets or {},
        namespace=namespace,
    )
    if volumes:
        kwargs["volumes"] = volumes

    return run_job(**kwargs)
