from __future__ import annotations

import shutil
import subprocess  # nosec B404 - local tooling wrapper invokes trusted Docker CLI
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_BASE = "ghcr.io/hadolint/hadolint"
IMAGE_SHA256 = "sha256:27086352fd5e1907ea2b934eb1023f217c5ae087992eb59fde121dce9c9ff21e"
IMAGE = f"{IMAGE_BASE}@{IMAGE_SHA256}"
IMAGE_PULL_REF = IMAGE
RUN_TIMEOUT_SECONDS = 90
PULL_TIMEOUT_SECONDS = 300
DOCKER_VERSION_TIMEOUT_SECONDS = 90
HADOLINT_ARGS = ["hadolint", "--ignore", "DL3013"]


def resolve_docker() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise SystemExit("Docker CLI is required for hadolint but was not found.")
    return docker


def run(
    command: list[str],
    *,
    timeout: int,
    description: str,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(  # nosec B603 - command is built from fixed args and filenames
            command,
            cwd=REPO_ROOT,
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise SystemExit(f"Timed out after {timeout}s while {description}.") from error


def ensure_image(docker: str) -> None:
    inspect = subprocess.run(  # nosec B603 - fixed local Docker inspect command
        [docker, "image", "inspect", IMAGE],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        return

    print(f"[hadolint] Pulling pinned image {IMAGE_PULL_REF}...", file=sys.stderr)
    run(
        [docker, "pull", IMAGE_PULL_REF],
        timeout=PULL_TIMEOUT_SECONDS,
        description=f"pulling {IMAGE_PULL_REF}",
    )


def lint_file(docker: str, filename: str) -> None:
    path = Path(filename)
    if not path.exists():
        raise SystemExit(f"Hadolint target does not exist: {filename}")

    container_name = f"hadolint-pre-commit-{uuid.uuid4().hex[:12]}"
    command = [
        docker,
        "run",
        "--name",
        container_name,
        "--rm",
        "-v",
        f"{REPO_ROOT}:/work",
        "-w",
        "/work",
        IMAGE,
        *HADOLINT_ARGS,
        filename,
    ]

    try:
        run(
            command,
            timeout=RUN_TIMEOUT_SECONDS,
            description=f"running hadolint on {filename}",
        )
    except SystemExit:
        subprocess.run(  # nosec B603 - best-effort cleanup for timed-out/failed container
            [docker, "rm", "-f", container_name],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raise


def main(argv: list[str]) -> int:
    if len(argv) <= 1:
        return 0

    docker = resolve_docker()
    run(
        [docker, "version"],
        timeout=DOCKER_VERSION_TIMEOUT_SECONDS,
        description="checking Docker availability",
    )
    ensure_image(docker)

    for filename in argv[1:]:
        lint_file(docker, filename)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
