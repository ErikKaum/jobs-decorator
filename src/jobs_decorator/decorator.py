"""The @job decorator — main entry point for the library."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, overload

from .dependencies import resolve_dependencies
from .handle import JobHandle
from .sidecar import Sidecar

F = TypeVar("F", bound=Callable[..., Any])


class JobFunction:
    """A decorated function that can be run locally or as a remote HF job."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        flavor: str,
        timeout: str,
        dependencies: Optional[list[str]],
        python: Optional[str],
        extras: Optional[list[str]],
        dependency_group: Optional[str],
        locked: bool,
        image: Optional[str],
        dockerfile: Optional[str],
        sidecar: Optional[Sidecar],
        volumes: Optional[list],
        env: Optional[dict[str, str]],
        secrets: Optional[dict[str, str]],
        namespace: Optional[str],
    ) -> None:
        self._fn = fn
        self._flavor = flavor
        self._timeout = timeout
        self._dependencies = dependencies
        self._python = python
        self._extras = extras
        self._dependency_group = dependency_group
        self._locked = locked
        self._image = image
        self._dockerfile = dockerfile
        self._sidecar = sidecar
        self._volumes = volumes
        self._env = env
        self._secrets = secrets
        self._namespace = namespace

        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the function locally (same as .local())."""
        return self.local(*args, **kwargs)

    def local(self, *args: Any, **kwargs: Any) -> Any:
        """Run the function locally."""
        return self._fn(*args, **kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> JobHandle:
        """Submit the function as a remote HF job.

        Returns a JobHandle for tracking progress and retrieving results.
        """
        from .backend import submit_job

        deps, python_requires = resolve_dependencies(
            dependencies=self._dependencies,
            extras=self._extras,
            dependency_group=self._dependency_group,
            locked=self._locked,
            python=self._python,
        )

        return submit_job(
            self._fn,
            args,
            kwargs,
            flavor=self._flavor,
            timeout=self._timeout,
            dependencies=deps,
            python_requires=python_requires,
            image=self._image,
            dockerfile=self._dockerfile,
            sidecar=self._sidecar,
            volumes=self._volumes,
            env=self._env,
            secrets=self._secrets,
            namespace=self._namespace,
        )


@overload
def job(fn: F) -> JobFunction: ...
@overload
def job(
    *,
    flavor: str = ...,
    timeout: str = ...,
    dependencies: Optional[list[str]] = ...,
    python: Optional[str] = ...,
    extras: Optional[list[str]] = ...,
    dependency_group: Optional[str] = ...,
    locked: bool = ...,
    image: Optional[str] = ...,
    dockerfile: Optional[str] = ...,
    sidecar: Optional[Sidecar] = ...,
    volumes: Optional[list] = ...,
    env: Optional[dict[str, str]] = ...,
    secrets: Optional[dict[str, str]] = ...,
    namespace: Optional[str] = ...,
) -> Callable[[F], JobFunction]: ...


def job(
    fn: Optional[F] = None,
    *,
    flavor: str = "cpu-basic",
    timeout: str = "30m",
    dependencies: Optional[list[str]] = None,
    python: Optional[str] = None,
    extras: Optional[list[str]] = None,
    dependency_group: Optional[str] = None,
    locked: bool = False,
    image: Optional[str] = None,
    dockerfile: Optional[str] = None,
    sidecar: Optional[Sidecar] = None,
    volumes: Optional[list] = None,
    env: Optional[dict[str, str]] = None,
    secrets: Optional[dict[str, str]] = None,
    namespace: Optional[str] = None,
) -> JobFunction | Callable[[F], JobFunction]:
    """Decorator that turns a Python function into a remote HF job.

    Can be used with or without arguments::

        @job
        def simple_task():
            ...

        @job(flavor="a10g-small", timeout="2h", dependencies=["torch"])
        def train(model_name: str):
            ...

    The decorated function gains ``.remote()`` and ``.local()`` methods:

    - ``train.remote("bert-base")`` — submits as an HF job, returns a JobHandle
    - ``train.local("bert-base")`` — runs locally (also the default for ``train()``)

    Parameters
    ----------
    flavor : str
        Hardware flavor. Default "cpu-basic".
    timeout : str
        Max runtime (e.g. "30m", "2h", "1d"). Default "30m".
    dependencies : list[str], optional
        Explicit pip dependencies. If omitted, reads from pyproject.toml.
    python : str, optional
        Python version constraint (e.g. ">=3.12"). Defaults to pinning the
        local Python version (e.g. "==3.10") so cloudpickle serialization
        is compatible between local and remote.
    extras : list[str], optional
        Optional dependency groups from pyproject.toml to include.
    dependency_group : str, optional
        PEP 735 dependency group to include.
    locked : bool
        If True, pin all versions from uv.lock.
    image : str, optional
        Docker image. Switches to run_job() backend.
    dockerfile : str, optional
        Path to Dockerfile. Switches to run_job() backend.
    sidecar : Sidecar, optional
        A service to run alongside the function (e.g. TEI, TGI). The sidecar
        process is started in the background inside the sidecar's container
        image, and the function runs after the service passes its readiness
        check. See :class:`Sidecar` for details.
    volumes : list[Volume], optional
        Volumes to mount in the job container. Use ``Volume`` from
        ``huggingface_hub`` (e.g.
        ``Volume(type="dataset", source="org/data", mount_path="/data")``).
    env : dict, optional
        Environment variables for the job.
    secrets : dict, optional
        Secret environment variables (encrypted).
    namespace : str, optional
        HF namespace (user or org) for the job.
    """

    def decorator(fn: F) -> JobFunction:
        return JobFunction(
            fn,
            flavor=flavor,
            timeout=timeout,
            dependencies=dependencies,
            python=python,
            extras=extras,
            dependency_group=dependency_group,
            locked=locked,
            image=image,
            dockerfile=dockerfile,
            sidecar=sidecar,
            volumes=volumes,
            env=env,
            secrets=secrets,
            namespace=namespace,
        )

    if fn is not None:
        return decorator(fn)
    return decorator
