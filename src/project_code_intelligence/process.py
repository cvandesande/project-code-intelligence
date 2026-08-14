"""Constrained subprocess helpers.

All project subprocess calls go through this module so command execution rules
are easy to audit: no shell, explicit timeouts at call sites, text mode, and
argument-vector commands only.
"""

from __future__ import annotations

import contextlib
import os
import shutil

# Centralized, shell-free subprocess boundary.
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import TYPE_CHECKING

from project_code_intelligence import config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

# Names of container engines we know how to drive. Order matters: docker is
# tried first because that's still the most common config, then podman as the
# drop-in replacement. Operators can pin via the env var below to skip the
# search entirely.
_CONTAINER_ENGINE_CANDIDATES = ("docker", "podman")
_CONTAINER_ENGINE_ENV_VAR = "PCI_CONTAINER_ENGINE"

PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT
DEVNULL = subprocess.DEVNULL
CalledProcessError = subprocess.CalledProcessError
CompletedProcess = subprocess.CompletedProcess
SubprocessError = subprocess.SubprocessError
TimeoutExpired = subprocess.TimeoutExpired


@dataclass(frozen=True)
class RunOptions:
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    input_text: str | None = None
    capture_output: bool = False
    stdout: int | TextIO | None = None
    stderr: int | TextIO | None = None
    timeout: float | None = None
    check: bool = False


@dataclass(frozen=True)
class PopenOptions:
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    stdout: int | TextIO | None = None
    stderr: int | TextIO | None = None
    stdin: int | TextIO | None = None
    start_new_session: bool = False


def run(command: Sequence[str], options: RunOptions | None = None) -> subprocess.CompletedProcess[str]:
    options = options or RunOptions()
    if not command:
        msg = "subprocess command must not be empty"
        raise ValueError(msg)
    if any(not part for part in command):
        msg = "subprocess command arguments must be non-empty strings"
        raise ValueError(msg)
    # shell=False and validated argv; callers supply fixed commands or trusted config.
    return subprocess.run(  # nosec B603
        list(command),
        cwd=options.cwd,
        env=dict(options.env) if options.env is not None else None,
        input=options.input_text,
        text=True,
        capture_output=options.capture_output,
        stdout=options.stdout,
        stderr=options.stderr,
        timeout=options.timeout,
        check=options.check,
        shell=False,
    )


def popen(command: Sequence[str], options: PopenOptions | None = None) -> subprocess.Popen[str]:
    opts = options or PopenOptions()
    if not command:
        msg = "subprocess command must not be empty"
        raise ValueError(msg)
    if any(not part for part in command):
        msg = "subprocess command arguments must be non-empty strings"
        raise ValueError(msg)
    return subprocess.Popen(  # nosec B603
        list(command),
        cwd=opts.cwd,
        env=dict(opts.env) if opts.env is not None else None,
        stdout=opts.stdout,
        stderr=opts.stderr,
        stdin=opts.stdin,
        text=True,
        start_new_session=opts.start_new_session,
        shell=False,
    )


# === Container engine detection ===========================================
#
# We accept docker or podman as the container engine — they share enough CLI
# surface (`<engine> compose`, `<engine> info`, etc.) for the things we drive.
# A single detector keeps every callsite consistent and lets operators pin
# via the env var.


def container_engine_path() -> str | None:
    """Return the absolute path of the configured container engine, or None if absent.

    Not cached: shutil.which is cheap and callers may legitimately re-probe (tests,
    or environments where PATH changes mid-run). If you need a stable handle within
    a single high-frequency loop, capture the result in a local.
    """
    override = os.environ.get(_CONTAINER_ENGINE_ENV_VAR, "").strip()
    if override:
        return shutil.which(override)
    for candidate in _CONTAINER_ENGINE_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    return None


def container_engine_name() -> str:
    """Return the basename of the resolved engine ('docker' or 'podman'), or 'docker' as a label fallback."""
    path = container_engine_path()
    if path is None:
        # Fall back to 'docker' purely for display in error messages.
        return "docker"
    return Path(path).name


# === Docker / Podman sandbox helper ========================================
#
# All container-engine invocations go through `run_docker` so we have one
# place that (a) restricts the executable to a known engine, (b) resolves it
# to an absolute path so PATH manipulation can't swap the binary, and (c)
# rejects argument shapes known to escalate privileges or expose the host.

_DOCKER_ARG_BLOCKLIST = {
    "--privileged",
    "--cap-add=ALL",
    "--security-opt=label=disable",
    "--pid=host",
    "--ipc=host",
    "--network=host",
    "--userns=host",
}
_DOCKER_DANGEROUS_VOLUME_PREFIXES = (
    "/:",
    "/etc:",
    "/var:",
    "/root:",
    "/home:",
    "/dev:",
    "/proc:",
    "/sys:",
)


def _argument_is_blocked(arg: str) -> bool:
    if arg in _DOCKER_ARG_BLOCKLIST:
        return True
    if arg.startswith(("-v=", "--volume=")):
        value = arg.split("=", 1)[1]
        if value.startswith(_DOCKER_DANGEROUS_VOLUME_PREFIXES):
            return True
    return False


def _audit_container_engine_args(args: Sequence[str], *, caller: str) -> None:
    saw_volume_flag = False
    for arg in args:
        if _argument_is_blocked(arg):
            raise ValueError(f"container engine argument refused by {caller}: {arg!r}")
        if saw_volume_flag and arg.startswith(_DOCKER_DANGEROUS_VOLUME_PREFIXES):
            raise ValueError(f"container engine volume mount refused by {caller}: {arg!r}")
        saw_volume_flag = arg in {"-v", "--volume"}


_COMPOSE_FILE_ENV_VAR = "PCI_COMPOSE_FILE"
_PACKAGE_NAME = "project_code_intelligence"
_COMPOSE_RESOURCE = "docker-compose.yml"
_COMPOSE_PROJECT_DIR_ENV_VAR = "PCI_COMPOSE_PROJECT_DIR"
_COMPOSE_CACHE_DIR_ENV_VAR = "PCI_COMPOSE_CACHE_DIR"
_APP_CACHE_DIR_NAME = "project-code-intelligence"
_INIT_EXTENSIONS_RESOURCE = Path("docker/pgvector/init-extensions.sql")
_COMPOSE_CONTEXT_PACKAGE_FILES = (
    (Path("docker/build-context/pyproject.toml"), Path("pyproject.toml")),
    (Path("docker/build-context/README.md"), Path("README.md")),
    (Path("docker/build-context/LICENSE"), Path("LICENSE")),
    (Path("docker/fastembed/Dockerfile"), Path("docker/fastembed/Dockerfile")),
    (Path("docker/llamacpp-rocm/Dockerfile"), Path("docker/llamacpp-rocm/Dockerfile")),
    (Path("docker/llamacpp-rocm/entrypoint.sh"), Path("docker/llamacpp-rocm/entrypoint.sh")),
    (Path("docker/llamacpp-cuda/Dockerfile"), Path("docker/llamacpp-cuda/Dockerfile")),
    (Path("docker/llamacpp-cuda/entrypoint.sh"), Path("docker/llamacpp-cuda/entrypoint.sh")),
    (Path("scripts/select_llamacpp_rocm_bundle.py"), Path("scripts/select_llamacpp_rocm_bundle.py")),
    (Path("bin/pci-embedding-server"), Path("pci-embedding-server")),
)
_COMPOSE_CONTEXT_PACKAGE_SOURCE_FILES = (
    Path("__init__.py"),
    Path("common.py"),
    Path("config.py"),
    Path("exceptions.py"),
    Path("http_client.py"),
    Path("process.py"),
    Path("rocm_bundles.py"),
    Path("runtime.py"),
    Path("embedding/__init__.py"),
    Path("embedding/fastembed_server.py"),
    Path("embedding/http_common.py"),
)


def _cache_root() -> Path:
    override = os.environ.get(_COMPOSE_CACHE_DIR_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / _APP_CACHE_DIR_NAME
    home = os.environ.get("HOME", "").strip()
    if home:
        if sys.platform == "darwin":
            return Path(home).expanduser() / "Library" / "Caches" / _APP_CACHE_DIR_NAME
        return Path(home).expanduser() / ".cache" / _APP_CACHE_DIR_NAME
    return Path(tempfile.gettempdir()) / _APP_CACHE_DIR_NAME


def compose_cache_dir() -> Path:
    """Return the directory used for materialized bundled Compose assets."""
    return _cache_root()


def _chmod_quiet(path: Path, mode: int) -> None:
    """Best-effort chmod; ownership/filesystem restrictions are not fatal here."""
    with contextlib.suppress(OSError):
        path.chmod(mode)


def _write_text_if_changed(path: Path, content: str) -> bool:
    """Write content to path if it differs from the current content. Returns True if changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_quiet(path.parent, 0o700)
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except OSError:
        pass
    if path.exists():
        _chmod_quiet(path, 0o600)
    _ = path.write_text(content, encoding="utf-8")
    _chmod_quiet(path, 0o600)
    return True


def _copy_file_if_changed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _chmod_quiet(target.parent, 0o700)
    try:
        if target.exists() and target.read_bytes() == source.read_bytes():
            return
    except OSError:
        pass
    if target.exists():
        _chmod_quiet(target, 0o600)
    _ = shutil.copyfile(source, target)
    _chmod_quiet(target, 0o600)


def _remove_generated_tree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError:
        pass
    else:
        return
    # Recovery path: restrictive permissions blocked deletion. Make the tree
    # owner-writable and retry once; a second failure propagates.
    for current_dir, dir_names, file_names in os.walk(path):
        current_path = Path(current_dir)
        _chmod_quiet(current_path, 0o700)
        for name in dir_names:
            _chmod_quiet(current_path / name, 0o700)
        for name in file_names:
            _chmod_quiet(current_path / name, 0o600)
    shutil.rmtree(path)


def _copy_minimal_package_source(package_dir: Path, target_dir: Path) -> None:
    _remove_generated_tree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    for relative_path in _COMPOSE_CONTEXT_PACKAGE_SOURCE_FILES:
        source = package_dir / relative_path
        if source.is_file():
            _copy_file_if_changed(source, target_dir / relative_path)


def _copy_package_context_assets(package_dir: Path, context_dir: Path) -> None:
    for package_relative, context_relative in _COMPOSE_CONTEXT_PACKAGE_FILES:
        source = package_dir / package_relative
        if source.is_file():
            _copy_file_if_changed(source, context_dir / context_relative)


def _package_dir() -> Path | None:
    try:
        ref = importlib_resources.files(_PACKAGE_NAME)
    except (ModuleNotFoundError, TypeError, AttributeError):
        return None
    path = Path(str(ref))
    return path if path.exists() else None


def _source_checkout_root(package_dir: Path) -> Path | None:
    candidate = package_dir.parent.parent
    if (candidate / "pyproject.toml").is_file() and (candidate / "docker-compose.yml").is_file():
        return candidate
    return None


def _materialize_installed_project_context(package_dir: Path, context_dir: Path) -> Path:
    _copy_minimal_package_source(package_dir, context_dir / "src" / _PACKAGE_NAME)
    _copy_package_context_assets(package_dir, context_dir)
    (context_dir / "models").mkdir(parents=True, exist_ok=True)
    return context_dir


def _project_dir_path(package_dir: Path) -> tuple[Path, bool]:
    """Resolve the project directory path without materializing anything.

    Returns (path, needs_materialization); needs_materialization is True
    only for the installed-package cache dir, whose contents (minimal
    source subset, Dockerfiles, build-context pyproject.toml, ...) must be
    staged before the directory is actually usable as a build/run context.
    Pure path resolution, safe to call from read-only diagnostics.
    """
    override_project_dir = os.environ.get(_COMPOSE_PROJECT_DIR_ENV_VAR, "").strip()
    if override_project_dir:
        return Path(override_project_dir).expanduser(), False
    if source_root := _source_checkout_root(package_dir):
        return source_root, False
    return _cache_root() / "compose-context", True


def _resolve_project_dir(package_dir: Path) -> Path:
    """Resolve the project directory used for Compose's --project-directory and Quadlet volume mounts.

    Resolution order: PCI_COMPOSE_PROJECT_DIR override, the source checkout
    root when running from one, or a minimal materialized context staged
    from installed package data. Materializes that staged context when
    needed -- for a path-only lookup with no file I/O side effects, use
    _project_dir_path. Not for build contexts -- see
    _resolve_build_context_dir.
    """
    project_dir, needs_materialization = _project_dir_path(package_dir)
    if needs_materialization:
        return _materialize_installed_project_context(package_dir, project_dir)
    return project_dir


def models_dir() -> Path:
    """Resolve the models/ directory GPU embedding backends read/download GGUF files from.

    Same directory Quadlet's llama-rocm/llama-cuda .container units
    bind-mount as /models (see materialize_quadlet_units). Read-only: does
    not materialize the installed-package cache dir as a side effect, so
    it's safe to call from diagnostics (`pci doctor`) that shouldn't do
    file I/O just from being run.
    """
    package_dir = _package_dir()
    if package_dir is None:
        return Path("models")
    project_dir, _ = _project_dir_path(package_dir)
    return project_dir / "models"


def _resolve_build_context_dir(package_dir: Path) -> Path:
    """Resolve the directory a `.build` unit uses as its Podman build context.

    Always the staged context, even from a source checkout: the Dockerfiles
    `COPY pyproject.toml ...` from the build context root, and only
    docker/build-context/pyproject.toml registers the container-specific
    entry points (e.g. pci-fastembed-server) -- the real repository root
    pyproject.toml only registers the consolidated `pci` binary. Using the
    raw checkout root here would copy the wrong pyproject.toml into the
    image and leave the container's ENTRYPOINT unresolvable.
    """
    override_project_dir = os.environ.get(_COMPOSE_PROJECT_DIR_ENV_VAR, "").strip()
    if override_project_dir:
        return Path(override_project_dir).expanduser()
    return _materialize_installed_project_context(package_dir, _cache_root() / "compose-context")


def _materialize_compose_file(package_dir: Path) -> tuple[Path, Path]:
    project_dir = _resolve_project_dir(package_dir)

    source_compose = package_dir / _COMPOSE_RESOURCE
    compose_text = source_compose.read_text(encoding="utf-8")
    init_extensions = project_dir / _INIT_EXTENSIONS_RESOURCE
    if not init_extensions.is_file():
        package_init_extensions = package_dir / _INIT_EXTENSIONS_RESOURCE
        if package_init_extensions.is_file():
            _copy_file_if_changed(package_init_extensions, init_extensions)
    if init_extensions.is_file():
        compose_text = compose_text.replace(f"./{_INIT_EXTENSIONS_RESOURCE.as_posix()}", str(init_extensions))

    materialized = _cache_root() / "docker-compose.yml"
    _ = _write_text_if_changed(materialized, compose_text)
    return materialized, project_dir


def compose_file_args() -> list[str]:
    """Return ["-f", "/path/to/docker-compose.yml"] for use in compose subcommands.

    Resolution order:
    1. PCI_COMPOSE_FILE environment variable — use this to
       point at a customised compose file without modifying the installed package.
    2. Materialized bundled docker-compose.yml from installed package data.
    3. Empty list — callers degrade to CWD-based discovery.
    """
    override = os.environ.get(_COMPOSE_FILE_ENV_VAR, "").strip()
    if override:
        return ["-f", override]
    package_dir = _package_dir()
    if package_dir is not None and (package_dir / _COMPOSE_RESOURCE).is_file():
        compose_path, project_dir = _materialize_compose_file(package_dir)
        return ["--project-directory", str(project_dir), "-f", str(compose_path)]
    return []


def run_docker(args: Sequence[str], options: RunOptions | None = None) -> subprocess.CompletedProcess[str]:
    """Run a docker- or podman-compatible subprocess with arg sandbox and absolute-path resolution."""
    if not args:
        msg = "container engine invocation must include at least one subcommand"
        raise ValueError(msg)
    resolved = container_engine_path()
    if resolved is None:
        msg = "no container engine (docker or podman) found on PATH"
        raise FileNotFoundError(msg)
    _audit_container_engine_args(args, caller="run_docker")
    return run([resolved, *args], options)


# === Podman (Quadlet) engine ===============================================
#
# Quadlet-managed containers (the embedding backends) always run under
# Podman -- Quadlet is a Podman-only systemd generator, unlike the compose
# path above which stays docker-preferred for pgvector. Keeping this
# resolver separate matters: on a host with both engines installed,
# resolving generically and running `docker ps` would not see
# Podman-managed Quadlet containers.

_PODMAN_ENGINE = "podman"


def podman_path() -> str | None:
    """Return the absolute path to podman, or None if it is not on PATH."""
    return shutil.which(_PODMAN_ENGINE)


def run_podman(args: Sequence[str], options: RunOptions | None = None) -> subprocess.CompletedProcess[str]:
    """Run a podman subprocess with the same arg sandbox as run_docker."""
    if not args:
        msg = "podman invocation must include at least one subcommand"
        raise ValueError(msg)
    resolved = podman_path()
    if resolved is None:
        msg = "podman not found on PATH"
        raise FileNotFoundError(msg)
    _audit_container_engine_args(args, caller="run_podman")
    return run([resolved, *args], options)


# === systemd --user ==========================================================
#
# All Quadlet-managed embedding services are started, stopped, and reloaded
# through `systemctl --user`, never invoked as `podman run`/`podman build`
# directly -- Quadlet's generated units own that. Centralized here for the
# same auditability reason `run_docker`/`run_podman` are centralized.

_SYSTEMCTL_USER_ALLOWED_SUBCOMMANDS = frozenset({
    "start",
    "stop",
    "restart",
    "daemon-reload",
    "is-active",
    "status",
    "show",
})


def run_systemctl_user(args: Sequence[str], options: RunOptions | None = None) -> subprocess.CompletedProcess[str]:
    """Run `systemctl --user <args>` through the sandboxed run() helper."""
    if not args or args[0] not in _SYSTEMCTL_USER_ALLOWED_SUBCOMMANDS:
        subcommand = args[0] if args else "<empty>"
        msg = f"systemctl --user subcommand not allowed: {subcommand!r}"
        raise ValueError(msg)
    resolved = shutil.which("systemctl")
    if resolved is None:
        msg = "systemctl not found on PATH"
        raise FileNotFoundError(msg)
    return run([resolved, "--user", *args], options)


# === Quadlet unit materialization ===========================================
#
# Quadlet unit files are static INI text -- unlike Compose, there is no
# ${VAR:-default} expansion at "start" time. The bundled quadlet/*.container,
# *.build, and *.volume templates carry @TOKEN@ placeholders (the same idea
# as the one substitution _materialize_compose_file does for
# init-extensions.sql, just applied to every parameterized field) that get
# resolved here from the same PCI_* env vars and defaults docker-compose.yml
# used for these services before they moved to Quadlet.

_QUADLET_UNIT_DIR_ENV_VAR = "PCI_QUADLET_UNIT_DIR"
_QUADLET_RESOURCE_DIR = Path("quadlet")

QUADLET_UNIT_FILES = (
    "pci-fastembed.build",
    "pci-fastembed-models.volume",
    "pci-fastembed.container",
    "pci-lemonade-huggingface.volume",
    "pci-lemonade-llama.volume",
    "pci-lemonade-cache.volume",
    "pci-lemonade-npu.container",
    "pci-llama-rocm.build",
    "pci-llamacpp-rocm.volume",
    "pci-llama-rocm.container",
    "pci-llama-cuda.build",
    "pci-llama-cuda.container",
)


def quadlet_unit_dir() -> Path:
    """Return the directory Quadlet unit files are materialized into.

    Resolution order: PCI_QUADLET_UNIT_DIR override, else the rootless
    Quadlet search path ($XDG_CONFIG_HOME/containers/systemd, falling back
    to ~/.config/containers/systemd for a user running `systemctl --user`).
    """
    override = os.environ.get(_QUADLET_UNIT_DIR_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / "containers" / "systemd"
    home = os.environ.get("HOME", "").strip()
    if home:
        return Path(home).expanduser() / ".config" / "containers" / "systemd"
    return Path(tempfile.gettempdir()) / "containers" / "systemd"


def _environment_block(pairs: Sequence[tuple[str, str | None]]) -> str:
    return "\n".join(f"Environment={key}={value}" for key, value in pairs if value is not None)


def _fastembed_substitutions(env: config.Env) -> dict[str, str]:
    model = config.env_text("PCI_FASTEMBED_MODEL", config.DEFAULT_FASTEMBED_MODEL, env=env)
    host = config.env_text("PCI_BIND_HOST", "127.0.0.1", env=env)
    port = config.env_text("PCI_EMBEDDING_PORT", "18081", env=env)
    return {
        "@FASTEMBED_ENVIRONMENT@": _environment_block([
            ("PCI_FASTEMBED_MODEL", model),
            ("PCI_FASTEMBED_CACHE_DIR", "/models/fastembed"),
            # Bind-all inside the container network namespace only; the host
            # exposure is PublishPort=, which honors PCI_BIND_HOST below.
            ("PCI_FASTEMBED_HOST", "0.0.0.0"),  # noqa: S104  # nosec B104
            ("PCI_FASTEMBED_PORT", "18081"),
        ]),
        "@FASTEMBED_PUBLISH_PORT@": f"{host}:{port}:18081",
    }


def _lemonade_substitutions(env: config.Env) -> dict[str, str]:
    image = config.env_text("PCI_LEMONADE_IMAGE", "ghcr.io/lemonade-sdk/lemonade-server:latest", env=env)
    hf_token = config.env_text("HF_TOKEN", "", env=env)
    model = config.env_text("PCI_LEMONADE_EMBEDDING_MODEL", config.DEFAULT_LEMONADE_EMBEDDING_MODEL, env=env)
    host = config.env_text("PCI_BIND_HOST", "127.0.0.1", env=env)
    port = config.env_text("PCI_EMBEDDING_PORT", "18081", env=env)
    return {
        "@PCI_LEMONADE_IMAGE@": image or "",
        "@LEMONADE_ENVIRONMENT@": _environment_block([
            ("HF_TOKEN", hf_token),
            ("PCI_EMBEDDING_ENDPOINT_MODEL", model),
        ]),
        "@LEMONADE_PUBLISH_PORT@": f"{host}:{port}:13305",
    }


def _llama_server_environment_pairs(env: config.Env) -> list[tuple[str, str | None]]:
    hf_model_repo = config.env_text("PCI_HF_MODEL_REPO", "Qwen/Qwen3-Embedding-0.6B-GGUF", env=env)
    hf_model_file = config.env_text("PCI_HF_MODEL_FILE", config.DEFAULT_GPU_EMBEDDING_MODEL, env=env)
    llama_model = config.env_text("PCI_LLAMA_MODEL", f"/models/{config.DEFAULT_GPU_EMBEDDING_MODEL}", env=env)
    hf_token = config.env_text("HF_TOKEN", "", env=env)
    return [
        ("PCI_HF_MODEL_REPO", hf_model_repo),
        ("PCI_HF_MODEL_FILE", hf_model_file),
        ("PCI_LLAMA_MODEL", llama_model),
        # Bind-all inside the container network namespace only; the host exposure
        # is PublishPort=, which honors PCI_BIND_HOST below.
        ("PCI_EMBEDDING_HOST", "0.0.0.0"),  # noqa: S104  # nosec B104
        ("PCI_EMBEDDING_PORT", "18081"),
        ("PCI_LLAMA_SERVER_CTX", config.env_text("PCI_LLAMA_SERVER_CTX", "40960", env=env)),
        ("PCI_LLAMA_SERVER_BATCH", config.env_text("PCI_LLAMA_SERVER_BATCH", "2048", env=env)),
        ("PCI_LLAMA_SERVER_UBATCH", config.env_text("PCI_LLAMA_SERVER_UBATCH", "1024", env=env)),
        ("PCI_LLAMA_SERVER_PARALLEL", config.env_text("PCI_LLAMA_SERVER_PARALLEL", "4", env=env)),
        ("PCI_LLAMA_SERVER_N_GPU_LAYERS", config.env_text("PCI_LLAMA_SERVER_N_GPU_LAYERS", "999", env=env)),
        ("PCI_LLAMA_SERVER_EXTRA_ARGS", config.env_text("PCI_LLAMA_SERVER_EXTRA_ARGS", "", env=env)),
        ("HF_TOKEN", hf_token),
    ]


def _rocm_substitutions(env: config.Env) -> dict[str, str]:
    rocm_base_image = config.env_text("PCI_ROCM_BASE_IMAGE", "debian:stable-slim", env=env)
    host = config.env_text("PCI_BIND_HOST", "127.0.0.1", env=env)
    port = config.env_text("PCI_EMBEDDING_PORT", "18081", env=env)
    pairs = [
        ("PCI_AMD_GFX", config.env_text("PCI_AMD_GFX", "", env=env)),
        ("PCI_LLAMA_CPP_ROCM_BUNDLE", config.env_text("PCI_LLAMA_CPP_ROCM_BUNDLE", "", env=env)),
        ("PCI_LLAMA_CPP_ROCM_RELEASE", config.env_text("PCI_LLAMA_CPP_ROCM_RELEASE", "latest", env=env)),
        (
            "PCI_LLAMA_CPP_ROCM_REPO",
            config.env_text("PCI_LLAMA_CPP_ROCM_REPO", "lemonade-sdk/llamacpp-rocm", env=env),
        ),
        *_llama_server_environment_pairs(env),
    ]
    return {
        "@PCI_ROCM_BASE_IMAGE@": rocm_base_image or "",
        "@ROCM_ENVIRONMENT@": _environment_block(pairs),
        "@ROCM_PUBLISH_PORT@": f"{host}:{port}:18081",
    }


def _cuda_substitutions(env: config.Env) -> dict[str, str]:
    cuda_base_image = config.env_text("PCI_CUDA_BASE_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda", env=env)
    host = config.env_text("PCI_BIND_HOST", "127.0.0.1", env=env)
    port = config.env_text("PCI_EMBEDDING_PORT", "18081", env=env)
    pairs = [
        *_llama_server_environment_pairs(env),
        ("NVIDIA_VISIBLE_DEVICES", config.env_text("NVIDIA_VISIBLE_DEVICES", "all", env=env)),
        ("NVIDIA_DRIVER_CAPABILITIES", config.env_text("NVIDIA_DRIVER_CAPABILITIES", "compute,utility", env=env)),
    ]
    return {
        "@PCI_CUDA_BASE_IMAGE@": cuda_base_image or "",
        "@CUDA_ENVIRONMENT@": _environment_block(pairs),
        "@CUDA_PUBLISH_PORT@": f"{host}:{port}:18081",
    }


def materialize_quadlet_units(env: config.Env | None = None) -> list[Path]:
    """Render and write the bundled Quadlet unit templates for the embedding backends.

    Returns the paths of files that were newly written or changed; the
    caller should run `systemctl --user daemon-reload` when this is
    non-empty. Requires bundled quadlet/ package data, which is always
    present when this code is running from an installed or checked-out
    copy of the package.
    """
    resolved_env = os.environ if env is None else env
    package_dir = _package_dir()
    if package_dir is None or not (package_dir / _QUADLET_RESOURCE_DIR).is_dir():
        msg = "bundled quadlet/ unit templates not found in package data"
        raise FileNotFoundError(msg)
    source_dir = package_dir / _QUADLET_RESOURCE_DIR
    project_dir = _resolve_project_dir(package_dir)
    build_context_dir = _resolve_build_context_dir(package_dir)
    # Unlike Compose, Podman does not auto-create a missing bind-mount
    # source directory (llama-rocm/llama-cuda's ./models mount) -- it fails
    # the container instead. Mirror Compose's behavior here.
    (project_dir / "models").mkdir(parents=True, exist_ok=True)

    substitutions: dict[str, str] = {
        "@PROJECT_DIR@": str(project_dir),
        "@BUILD_CONTEXT_DIR@": str(build_context_dir),
    }
    substitutions.update(_fastembed_substitutions(resolved_env))
    substitutions.update(_lemonade_substitutions(resolved_env))
    substitutions.update(_rocm_substitutions(resolved_env))
    substitutions.update(_cuda_substitutions(resolved_env))

    unit_dir = quadlet_unit_dir()
    changed: list[Path] = []
    for filename in QUADLET_UNIT_FILES:
        content = (source_dir / filename).read_text(encoding="utf-8")
        for token, value in substitutions.items():
            content = content.replace(token, value)
        target = unit_dir / filename
        if _write_text_if_changed(target, content):
            changed.append(target)
    return changed
