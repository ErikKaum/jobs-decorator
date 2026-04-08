"""JobHandle — rich wrapper around a running HF job."""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from huggingface_hub import HfApi


# Terminal stages where the job is no longer running
_TERMINAL_STAGES = frozenset({"COMPLETED", "ERROR", "CANCELED"})


@dataclass
class JobHandle:
    """A handle to a running (or completed) HF job.

    Returned by ``decorated_fn.remote()``. Provides methods to inspect,
    wait for, and retrieve results from the job.
    """

    job_id: str
    url: str
    result_repo_id: str
    result_filename: str
    _api: HfApi = field(default_factory=HfApi, repr=False)

    def status(self) -> str:
        """Return the current job stage (RUNNING, COMPLETED, ERROR, etc.)."""
        info = self._api.inspect_job(job_id=self.job_id)
        return info.status.stage

    def logs(self) -> Iterator[str]:
        """Iterate over log lines from the job."""
        return self._api.fetch_job_logs(job_id=self.job_id)

    def metrics(self) -> dict[str, Any]:
        """Fetch current resource usage metrics (CPU, memory, GPU)."""
        return self._api.fetch_job_metrics(job_id=self.job_id)

    def cancel(self) -> None:
        """Cancel the running job."""
        self._api.cancel_job(job_id=self.job_id)

    def wait(self, poll_interval: float = 15.0, timeout: Optional[float] = None) -> str:
        """Block until the job reaches a terminal state.

        Returns the final stage string.
        """
        start = time.monotonic()
        while True:
            stage = self.status()
            if stage in _TERMINAL_STAGES:
                return stage
            if timeout is not None and (time.monotonic() - start) > timeout:
                raise TimeoutError(
                    f"Job {self.job_id} did not complete within {timeout}s "
                    f"(last stage: {stage})"
                )
            time.sleep(poll_interval)

    def result(self, poll_interval: float = 15.0, timeout: Optional[float] = None) -> Any:
        """Wait for the job to finish and return the deserialized return value.

        Raises ``RuntimeError`` if the job failed.
        """
        import cloudpickle

        final_stage = self.wait(poll_interval=poll_interval, timeout=timeout)

        # Check for error artifact first
        error_path = f"{self.result_filename}.error"
        try:
            error_bytes = self._api.hf_hub_download(
                repo_id=self.result_repo_id,
                filename=error_path,
                repo_type="dataset",
            )
            with open(error_bytes, "r") as f:
                error_data = json.load(f)
            raise RuntimeError(
                f"Remote job failed:\n{error_data['error']}"
            )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            # No error artifact — continue to fetch result

        if final_stage != "COMPLETED":
            raise RuntimeError(
                f"Job {self.job_id} ended with stage {final_stage}. "
                f"Check logs with handle.logs()"
            )

        # Download the result artifact
        result_path = self._api.hf_hub_download(
            repo_id=self.result_repo_id,
            filename=self.result_filename,
            repo_type="dataset",
        )
        with open(result_path, "r") as f:
            result_b64 = f.read()

        result_bytes = base64.b64decode(result_b64)
        return cloudpickle.loads(result_bytes)
