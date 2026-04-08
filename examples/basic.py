"""Basic usage of the @job decorator."""

from jobs_decorator import job


@job
def add(a: int, b: int) -> int:
    return a + b

# Run locally (default behavior)
result = add(2, 3)
print(f"Local result: {result}")

# Run as a remote HF job
handle = add.remote(2, 3)
print(f"Job submitted: {handle.url}")

# Wait for the result
result = handle.result()
print(f"Remote result: {result}")
