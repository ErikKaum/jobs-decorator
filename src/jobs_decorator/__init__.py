"""jobs-decorator — run Python functions as remote Hugging Face jobs."""

from .decorator import job, JobFunction
from .handle import JobHandle
from .sidecar import Sidecar

__all__ = ["job", "JobFunction", "JobHandle", "Sidecar"]
