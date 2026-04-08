"""Dependency resolution: explicit lists, pyproject.toml, uv.lock."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional


def resolve_dependencies(
    *,
    dependencies: Optional[list[str]] = None,
    extras: Optional[list[str]] = None,
    dependency_group: Optional[str] = None,
    locked: bool = False,
    python: Optional[str] = None,
    project_dir: Optional[Path] = None,
) -> tuple[list[str], str]:
    """Resolve dependencies into a flat list + python version constraint.

    Returns (deps, python_requires) where deps is a list of PEP 508 strings
    and python_requires is e.g. ">=3.12".
    """
    project_dir = project_dir or _find_project_root()
    # Pin to the local Python version by default so cloudpickle bytecode
    # serialized locally can be deserialized on the remote.
    local_version = f"=={sys.version_info.major}.{sys.version_info.minor}"
    python_requires = python or local_version

    # Explicit dependencies take priority
    if dependencies is not None:
        return dependencies, python_requires

    # Read from pyproject.toml
    pyproject_path = project_dir / "pyproject.toml" if project_dir else None
    if pyproject_path and pyproject_path.exists():
        return _resolve_from_pyproject(
            pyproject_path,
            extras=extras,
            dependency_group=dependency_group,
            locked=locked,
            python_override=python_requires,
        )

    return [], python_requires


def _find_project_root() -> Optional[Path]:
    """Walk up from cwd looking for pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _resolve_from_pyproject(
    pyproject_path: Path,
    *,
    extras: Optional[list[str]] = None,
    dependency_group: Optional[str] = None,
    locked: bool = False,
    python_override: Optional[str] = None,
) -> tuple[list[str], str]:
    """Extract dependencies from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    python_requires = python_override or project.get("requires-python", ">=3.10")

    if locked:
        return _resolve_from_lockfile(pyproject_path.parent, python_requires)

    deps: list[str] = list(project.get("dependencies", []))

    # Add optional dependency groups (extras)
    if extras:
        optional_deps = project.get("optional-dependencies", {})
        for extra in extras:
            deps.extend(optional_deps.get(extra, []))

    # Add dependency groups (PEP 735 / uv style)
    if dependency_group:
        groups = data.get("dependency-groups", {})
        deps.extend(groups.get(dependency_group, []))

    return deps, python_requires


def _resolve_from_lockfile(
    project_dir: Path, python_requires: str
) -> tuple[list[str], str]:
    """Read pinned versions from uv.lock."""
    lock_path = project_dir / "uv.lock"
    if not lock_path.exists():
        raise FileNotFoundError(
            f"locked=True but no uv.lock found at {lock_path}. "
            "Run `uv lock` first."
        )

    # uv.lock is TOML — parse package entries and extract name==version pins
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(lock_path, "rb") as f:
        lock_data = tomllib.load(f)

    deps: list[str] = []
    for package in lock_data.get("package", []):
        name = package.get("name", "")
        version = package.get("version", "")
        # Skip the project's own package
        source = package.get("source", {})
        if source.get("editable") or source.get("virtual"):
            continue
        if name and version:
            deps.append(f"{name}=={version}")

    return deps, python_requires
