"""Use environment variables and secrets in a remote job."""

from jobs_decorator import job


@job(
    flavor="cpu-basic",
    timeout="15m",
    dependencies=["requests"],
    env={"BATCH_SIZE": "32", "LOG_LEVEL": "INFO"},
    secrets={"API_KEY": "hf_...your_key_here..."},
)
def fetch_and_process(url: str) -> dict:
    import os

    import requests

    api_key = os.environ["API_KEY"]
    batch_size = int(os.environ["BATCH_SIZE"])

    response = requests.get(url, headers={"Authorization": f"Bearer {api_key}"})
    data = response.json()

    return {
        "status": response.status_code,
        "items_processed": len(data.get("items", [])),
        "batch_size": batch_size,
    }


handle = fetch_and_process.remote("https://api.example.com/data")
print(f"Job: {handle.url}")

result = handle.result()
print(result)
