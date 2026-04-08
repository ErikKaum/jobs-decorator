"""Generate self-contained PEP 723 UV scripts from decorated functions."""

from __future__ import annotations

import base64


def generate_uv_script(
    *,
    pickled_fn: bytes,
    pickled_args: bytes,
    dependencies: list[str],
    python_requires: str,
    result_repo_id: str,
    result_filename: str,
) -> str:
    """Build a PEP 723 UV script that runs the serialized function remotely.

    The script:
    1. Deserializes the function and arguments
    2. Calls the function
    3. Serializes the return value
    4. Uploads it to a private HF repo for the caller to retrieve
    """
    all_deps = _ensure_deps(dependencies, ["cloudpickle", "huggingface_hub"])

    fn_b85 = base64.b85encode(pickled_fn).decode("ascii")
    args_b85 = base64.b85encode(pickled_args).decode("ascii")

    dep_lines = "".join(f'#   "{dep}",\n' for dep in all_deps)

    lines = [
        "# /// script",
        f'# requires-python = "{python_requires}"',
        "# dependencies = [",
    ]
    for dep in all_deps:
        lines.append(f'#   "{dep}",')
    lines += [
        "# ]",
        "# ///",
        "",
        "import base64",
        "import cloudpickle",
        "import traceback",
        "",
        "",
        "def _main():",
        f"    fn = cloudpickle.loads(base64.b85decode({fn_b85!r}))",
        f"    args, kwargs = cloudpickle.loads(base64.b85decode({args_b85!r}))",
        "",
        "    result = fn(*args, **kwargs)",
        "",
        "    result_bytes = cloudpickle.dumps(result)",
        '    result_b64 = base64.b64encode(result_bytes).decode("ascii")',
        "",
        "    from huggingface_hub import HfApi",
        "    api = HfApi()",
        "    api.upload_file(",
        '        path_or_fileobj=result_b64.encode("ascii"),',
        f'        path_in_repo="{result_filename}",',
        f'        repo_id="{result_repo_id}",',
        '        repo_type="dataset",',
        "    )",
        f'    print("Result uploaded to {result_repo_id}")',
        "",
        "",
        'if __name__ == "__main__":',
        "    try:",
        "        _main()",
        "    except Exception:",
        "        traceback.print_exc()",
        "",
        "        import json",
        '        error_payload = json.dumps({"error": traceback.format_exc()}).encode()',
        "        from huggingface_hub import HfApi",
        "        api = HfApi()",
        "        api.upload_file(",
        "            path_or_fileobj=error_payload,",
        f'            path_in_repo="{result_filename}.error",',
        f'            repo_id="{result_repo_id}",',
        '            repo_type="dataset",',
        "        )",
        "        raise",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate_docker_script(
    *,
    pickled_fn: bytes,
    pickled_args: bytes,
    result_repo_id: str,
    result_filename: str,
) -> str:
    """Generate a plain Python bootstrap for docker-based jobs.

    Assumes cloudpickle and huggingface_hub are already installed in the image
    (or the user's Dockerfile installs them).
    """
    fn_b85 = base64.b85encode(pickled_fn).decode("ascii")
    args_b85 = base64.b85encode(pickled_args).decode("ascii")

    lines = [
        "import base64",
        "import cloudpickle",
        "import traceback",
        "",
        "",
        "def _main():",
        f"    fn = cloudpickle.loads(base64.b85decode({fn_b85!r}))",
        f"    args, kwargs = cloudpickle.loads(base64.b85decode({args_b85!r}))",
        "",
        "    result = fn(*args, **kwargs)",
        "",
        "    result_bytes = cloudpickle.dumps(result)",
        '    result_b64 = base64.b64encode(result_bytes).decode("ascii")',
        "",
        "    from huggingface_hub import HfApi",
        "    api = HfApi()",
        "    api.upload_file(",
        '        path_or_fileobj=result_b64.encode("ascii"),',
        f'        path_in_repo="{result_filename}",',
        f'        repo_id="{result_repo_id}",',
        '        repo_type="dataset",',
        "    )",
        "",
        "",
        'if __name__ == "__main__":',
        "    try:",
        "        _main()",
        "    except Exception:",
        "        traceback.print_exc()",
        "",
        "        import json",
        '        error_payload = json.dumps({"error": traceback.format_exc()}).encode()',
        "        from huggingface_hub import HfApi",
        "        api = HfApi()",
        "        api.upload_file(",
        "            path_or_fileobj=error_payload,",
        f'            path_in_repo="{result_filename}.error",',
        f'            repo_id="{result_repo_id}",',
        '            repo_type="dataset",',
        "        )",
        "        raise",
        "",
    ]
    return "\n".join(lines) + "\n"


def _ensure_deps(deps: list[str], required: list[str]) -> list[str]:
    """Ensure required packages are in the dep list (by base name)."""
    existing_names = set()
    for dep in deps:
        name = dep.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("~")[0].strip()
        existing_names.add(name.lower().replace("-", "_"))

    result = list(deps)
    for req in required:
        if req.lower().replace("-", "_") not in existing_names:
            result.append(req)
    return result
