from huggingface_hub import Volume
from jobs_decorator import job

volume = Volume(type="dataset", source="HuggingFaceFW/fineweb", mount_path="/data")

@job(flavor="cpu-basic", volumes=[volume])
def show_dir():
    import os
    return os.listdir("/data")


# Run as a remote HF job
handle = show_dir.remote()
print(f"Job submitted: {handle.url}")
result = handle.result()
print(f"Remote result: {result}")