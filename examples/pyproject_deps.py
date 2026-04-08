"""Test that dependencies from pyproject.toml are resolved correctly on the remote.

When no `dependencies` argument is passed, jobs-decorator reads your
pyproject.toml automatically.
"""

from jobs_decorator import job


@job(flavor="cpu-basic", timeout="10m")
def check_deps():
    import cloudpickle
    return {"cloudpickle_version": cloudpickle.__version__}

# Explicit dependencies for when you need packages not in pyproject.toml
@job(flavor="a10g-small", timeout="1h", dependencies=["torch"])
def task_with_torch():
    import torch

    return {"cuda_available": torch.cuda.is_available()}

# handle = check_deps.remote()
# print(f"First job: {handle.url}")
# result = handle.result()
# print(f"First result: {result}")

handle = task_with_torch.remote()
print(f"Second job: {handle.url}")
result = handle.result()
print(f"Second result: {result}")
