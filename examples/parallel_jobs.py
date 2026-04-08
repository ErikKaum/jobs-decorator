"""Submit multiple jobs in parallel and collect results."""

from jobs_decorator import job


@job(flavor="cpu-basic", timeout="10m")
def evaluate(dataset_name: str, split: str = "test") -> dict:
    # Simulate an evaluation task
    import hashlib

    score = int(hashlib.md5(dataset_name.encode()).hexdigest(), 16) % 100 / 100
    return {"dataset": dataset_name, "split": split, "score": score}


datasets = ["imdb", "sst2", "mnli", "qnli", "rte"]

# Launch all jobs concurrently
handles = [evaluate.remote(ds) for ds in datasets]

print(f"Submitted {len(handles)} jobs:")
for ds, h in zip(datasets, handles):
    print(f"  {ds}: {h.url}")

# Collect results as they finish
for ds, h in zip(datasets, handles):
    result = h.result(timeout=600)
    print(f"{result['dataset']}: {result['score']:.2f}")
