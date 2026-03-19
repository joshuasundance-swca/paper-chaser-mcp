from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess  # nosec B404 - local validator shells out to trusted CLI tools
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from defusedxml import ElementTree as DefusedET

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_BICEP = REPO_ROOT / "infra" / "main.bicep"
PARAM_FILES = [
    REPO_ROOT / "infra" / "main.dev.bicepparam",
    REPO_ROOT / "infra" / "main.staging.bicepparam",
    REPO_ROOT / "infra" / "main.prod.bicepparam",
]
APIM_POLICY = REPO_ROOT / "infra" / "policies" / "scholar-search-policy.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate deployment assets for the Azure-hosted Scholar Search MCP "
            "service."
        )
    )
    parser.add_argument(
        "--skip-az",
        action="store_true",
        help="Skip Azure Bicep lint/build checks.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker build and container smoke tests.",
    )
    parser.add_argument(
        "--require-az",
        action="store_true",
        help="Fail if Azure CLI is unavailable instead of skipping Azure checks.",
    )
    parser.add_argument(
        "--require-docker",
        action="store_true",
        help="Fail if Docker is unavailable instead of skipping Docker checks.",
    )
    parser.add_argument(
        "--image-tag",
        default="scholar-search-mcp:validation",
        help="Local Docker image tag used for build and smoke tests.",
    )
    return parser.parse_args()


def run(command: list[str], *, description: str) -> subprocess.CompletedProcess[str]:
    print(f"[validate-deployment] {description}")
    return subprocess.run(  # nosec B603 - command is built from fixed args and resolved tool paths
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def resolve_command(command: str) -> str | None:
    candidates = [command]
    if sys.platform.startswith("win"):
        candidates.extend([f"{command}.cmd", f"{command}.exe"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def require_command(command: str, *, required: bool, label: str) -> str | None:
    resolved = resolve_command(command)
    if resolved:
        return resolved
    if required:
        raise SystemExit(f"Required tool '{command}' for {label} is not available.")
    print(f"[validate-deployment] Skipping {label}: '{command}' is not available.")
    return None


def validate_xml_policy() -> None:
    print("[validate-deployment] Validating APIM policy XML")
    DefusedET.parse(APIM_POLICY)


def validate_bicep(az_command: str) -> None:
    run(
        [az_command, "bicep", "lint", "--file", str(MAIN_BICEP)],
        description="Lint main.bicep",
    )
    run(
        [az_command, "bicep", "build", "--file", str(MAIN_BICEP), "--stdout"],
        description="Build main.bicep",
    )
    for param_file in PARAM_FILES:
        run(
            [
                az_command,
                "bicep",
                "build-params",
                "--file",
                str(param_file),
                "--stdout",
            ],
            description=f"Build params file {param_file.name}",
        )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict | None = None,
) -> tuple[int, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SystemExit(f"Unsupported URL scheme for validation request: {url}")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit(f"Validation requests must stay local, got: {url}")

    data = None
    request_headers = headers or {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310 - URL is validated above and used only for local smoke tests
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def wait_for_health(url: str) -> None:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            status, _ = request_json(url)
        except Exception:
            time.sleep(1)
            continue
        if status == 200:
            return
        time.sleep(1)
    raise SystemExit(f"Timed out waiting for container health at {url}")


def validate_image_config(docker_command: str, image_tag: str) -> None:
    inspect = run(
        [docker_command, "image", "inspect", image_tag],
        description=f"Inspect Docker image {image_tag}",
    )
    payload = json.loads(inspect.stdout)
    if not payload:
        raise SystemExit(f"Docker inspect returned no data for image {image_tag}.")

    config = payload[0].get("Config", {})
    user = str(config.get("User") or "").strip().lower()
    if user in {"", "0", "root"}:
        raise SystemExit(
            "Docker image must run as a non-root user. "
            f"Current image user is {config.get('User')!r}."
        )

    healthcheck = config.get("Healthcheck")
    if not healthcheck or not healthcheck.get("Test"):
        raise SystemExit("Docker image must define a HEALTHCHECK.")


def validate_docker(docker_command: str, image_tag: str) -> None:
    run(
        [docker_command, "build", "-t", image_tag, "."],
        description=f"Build Docker image {image_tag}",
    )
    validate_image_config(docker_command, image_tag)

    port = free_port()
    container_name = f"scholar-search-validation-{port}"
    allowed_origin = "https://apim.validation.local"
    run(
        [
            docker_command,
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"{port}:8080",
            "-e",
            f"SCHOLAR_SEARCH_ALLOWED_ORIGINS={allowed_origin}",
            "-e",
            "SCHOLAR_SEARCH_HTTP_AUTH_HEADER=x-backend-auth",
            "-e",
            "SCHOLAR_SEARCH_HTTP_AUTH_TOKEN=local-test-token",
            image_tag,
        ],
        description=f"Start validation container {container_name}",
    )

    health_url = f"http://127.0.0.1:{port}/healthz"
    mcp_url = f"http://127.0.0.1:{port}/mcp/"

    try:
        wait_for_health(health_url)

        status, _ = request_json(
            mcp_url,
            method="POST",
            headers={"Origin": "https://blocked.example"},
            body={},
        )
        if status != 403:
            raise SystemExit(
                f"Expected blocked-origin /mcp request to return 403, got {status}."
            )

        status, _ = request_json(
            mcp_url,
            method="POST",
            headers={"Origin": allowed_origin},
            body={},
        )
        if status != 401:
            raise SystemExit(
                "Expected missing-backend-auth /mcp request to return 401, "
                f"got {status}."
            )

        status, _ = request_json(
            mcp_url,
            method="POST",
            headers={
                "Accept": "application/json, text/event-stream",
                "Origin": allowed_origin,
                "X-Backend-Auth": "local-test-token",
            },
            body={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        if status in {401, 403}:
            raise SystemExit("Authenticated /mcp request unexpectedly returned 401.")
    finally:
        subprocess.run(  # nosec B603 - fixed local cleanup command for validation container
            [docker_command, "rm", "-f", container_name],
            cwd=REPO_ROOT,
        )


def main() -> None:
    args = parse_args()

    validate_xml_policy()

    az_command = None
    if not args.skip_az:
        az_command = require_command(
            "az",
            required=args.require_az,
            label="Azure validation",
        )
    if az_command:
        validate_bicep(az_command)

    docker_command = None
    if not args.skip_docker:
        docker_command = require_command(
            "docker",
            required=args.require_docker,
            label="Docker validation",
        )
    if docker_command:
        validate_docker(docker_command, args.image_tag)

    print("[validate-deployment] Deployment asset validation completed successfully")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, file=sys.stdout)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode) from error
