from __future__ import annotations

import argparse
import json
import re
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
APIM_POLICY = REPO_ROOT / "infra" / "policies" / "paper-chaser-policy.xml"
SERVER_METADATA = REPO_ROOT / "server.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Validate deployment assets for the Azure-hosted Paper Chaser MCP service.")
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
        default="paper-chaser-mcp:validation",
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


def _read_server_metadata() -> dict:
    if not SERVER_METADATA.exists():
        raise SystemExit("Missing server.json. Public MCP packaging requires registry metadata.")
    try:
        payload = json.loads(SERVER_METADATA.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid server.json JSON: {error}") from error

    if not isinstance(payload, dict):
        raise SystemExit("server.json must contain a top-level JSON object.")
    return payload


def _extract_oci_packages(payload: dict) -> list[dict]:
    packages = payload.get("packages")
    if not isinstance(packages, list):
        return []
    oci_packages: list[dict] = []
    for candidate in packages:
        if not isinstance(candidate, dict):
            continue
        package_type = candidate.get("registryType") or candidate.get("registry_type") or candidate.get("type")
        if isinstance(package_type, str) and package_type.lower() == "oci":
            oci_packages.append(candidate)
    return oci_packages


def _expected_ghcr_identifier(server_name: str, version: str) -> str:
    parsed_name = urllib.parse.unquote(server_name).strip()
    name_match = re.match(r"^io\.github\.([^/]+)/([^/]+)$", parsed_name)
    if name_match is None:
        raise SystemExit(
            "server.json name must follow io.github.<owner>/<server> so the public OCI package path can be derived."
        )
    owner, server = name_match.groups()
    return f"ghcr.io/{owner.lower()}/{server.lower()}:{version}"


def validate_server_metadata() -> dict[str, str]:
    payload = _read_server_metadata()

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SystemExit("server.json must define a non-empty string 'name'.")
    if not name.startswith("io.github."):
        raise SystemExit(
            "server.json name must use the public GitHub namespace form (io.github.username/* or io.github.orgname/*)."
        )
    if "/" not in name:
        raise SystemExit("server.json name must include a server identifier path.")

    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise SystemExit("server.json must define a non-empty string 'version'.")

    repository = payload.get("repository")
    repository_url = repository.get("url") if isinstance(repository, dict) else None
    if not isinstance(repository_url, str) or not repository_url.strip():
        raise SystemExit("server.json repository.url must be a non-empty GitHub repository URL.")

    oci_packages = _extract_oci_packages(payload)
    if not oci_packages:
        raise SystemExit(
            "server.json must include at least one OCI package entry (registryType/registry_type/type == 'oci')."
        )

    primary_package = oci_packages[0]
    package_identifier = primary_package.get("identifier")
    if not isinstance(package_identifier, str) or not package_identifier.strip():
        raise SystemExit("server.json OCI packages must define a non-empty string 'identifier'.")

    expected_identifier = _expected_ghcr_identifier(name, version)
    if package_identifier != expected_identifier:
        raise SystemExit(
            "server.json OCI package identifier must match the public server-name-backed "
            f"GHCR package path. Expected {expected_identifier!r}, got "
            f"{package_identifier!r}."
        )

    transport = primary_package.get("transport")
    transport_type = transport.get("type") if isinstance(transport, dict) else None
    if transport_type != "stdio":
        raise SystemExit("Public OCI MCP packages must declare transport.type='stdio' in server.json.")

    return {
        "name": name,
        "version": version,
        "source_url": repository_url,
        "package_identifier": package_identifier,
    }


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


def _parse_image_env(config: dict) -> dict[str, str]:
    parsed: dict[str, str] = {}
    raw_env = config.get("Env")
    if not isinstance(raw_env, list):
        return parsed
    for entry in raw_env:
        if not isinstance(entry, str):
            continue
        key, sep, value = entry.partition("=")
        if sep:
            parsed[key] = value
    return parsed


def validate_image_config(
    docker_command: str,
    image_tag: str,
    *,
    expected_server_name: str,
    expected_server_version: str,
    expected_source_url: str,
) -> None:
    inspect = run(
        [docker_command, "image", "inspect", image_tag],
        description=f"Inspect Docker image {image_tag}",
    )
    payload = json.loads(inspect.stdout)
    if not payload:
        raise SystemExit(f"Docker inspect returned no data for image {image_tag}.")

    config = payload[0].get("Config", {})
    labels = config.get("Labels") or {}
    if not isinstance(labels, dict):
        raise SystemExit("Docker image labels payload must be an object.")

    declared_server_name = labels.get("io.modelcontextprotocol.server.name")
    if declared_server_name != expected_server_name:
        raise SystemExit(
            "Docker image label io.modelcontextprotocol.server.name must match "
            f"server.json name. Expected {expected_server_name!r}, got "
            f"{declared_server_name!r}."
        )

    image_version = labels.get("org.opencontainers.image.version")
    if image_version != expected_server_version:
        raise SystemExit(
            "Docker image label org.opencontainers.image.version must match "
            f"server.json version. Expected {expected_server_version!r}, got "
            f"{image_version!r}."
        )

    image_source = labels.get("org.opencontainers.image.source")
    if image_source != expected_source_url:
        raise SystemExit(
            "Docker image label org.opencontainers.image.source must match "
            f"server.json repository.url. Expected {expected_source_url!r}, got "
            f"{image_source!r}."
        )

    entrypoint = config.get("Entrypoint")
    if not isinstance(entrypoint, list) or not entrypoint:
        raise SystemExit("Docker image must define a non-empty ENTRYPOINT.")

    env = _parse_image_env(config)
    configured_transport = env.get("PAPER_CHASER_TRANSPORT")
    if configured_transport and configured_transport.lower() != "stdio":
        raise SystemExit("Docker image default transport must be stdio when PAPER_CHASER_TRANSPORT is set.")

    user = str(config.get("User") or "").strip().lower()
    if user in {"", "0", "root"}:
        raise SystemExit(f"Docker image must run as a non-root user. Current image user is {config.get('User')!r}.")


def validate_docker(
    docker_command: str,
    image_tag: str,
    *,
    expected_server_name: str,
    expected_server_version: str,
    expected_source_url: str,
) -> None:
    build_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run(
        [
            docker_command,
            "build",
            "-t",
            image_tag,
            "--build-arg",
            f"VERSION={expected_server_version}",
            "--build-arg",
            "VCS_REF=local-validate",
            "--build-arg",
            f"BUILD_DATE={build_date}",
            ".",
        ],
        description=f"Build Docker image {image_tag}",
    )
    validate_image_config(
        docker_command,
        image_tag,
        expected_server_name=expected_server_name,
        expected_server_version=expected_server_version,
        expected_source_url=expected_source_url,
    )

    port = free_port()
    container_name = f"paper-chaser-validation-{port}"
    allowed_origin = "https://apim.validation.local"
    # Keep HTTP smoke tests explicit so the image can remain stdio-first by default.
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
            f"PAPER_CHASER_ALLOWED_ORIGINS={allowed_origin}",
            "-e",
            "PAPER_CHASER_HTTP_AUTH_HEADER=x-backend-auth",
            "-e",
            "PAPER_CHASER_HTTP_AUTH_TOKEN=local-test-token",
            image_tag,
            "deployment-http",
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
            raise SystemExit(f"Expected blocked-origin /mcp request to return 403, got {status}.")

        status, _ = request_json(
            mcp_url,
            method="POST",
            headers={"Origin": allowed_origin},
            body={},
        )
        if status != 401:
            raise SystemExit(f"Expected missing-backend-auth /mcp request to return 401, got {status}.")

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
    server_metadata = validate_server_metadata()

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
        validate_docker(
            docker_command,
            args.image_tag,
            expected_server_name=server_metadata["name"],
            expected_server_version=server_metadata["version"],
            expected_source_url=server_metadata["source_url"],
        )

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
