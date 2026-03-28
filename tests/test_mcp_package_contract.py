from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
SERVER_JSON = REPO_ROOT / "server.json"
PYPROJECT = REPO_ROOT / "pyproject.toml"

SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?(?:\+[0-9A-Za-z][0-9A-Za-z.-]*)?$")


def _read_server_json() -> dict[str, Any]:
    assert SERVER_JSON.exists(), "Missing server.json MCP metadata."
    payload = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "server.json must contain a JSON object."
    return payload


def _iter_dicts(value: Any) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    if isinstance(value, dict):
        discovered.append(value)
        for item in value.values():
            discovered.extend(_iter_dicts(item))
        return discovered
    if isinstance(value, list):
        for item in value:
            discovered.extend(_iter_dicts(item))
    return discovered


def _read_pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^\[project\]\s+.*?^version\s*=\s*\"(?P<version>[^\"]+)\"",
        text,
    )
    assert match is not None, "pyproject.toml must define [project].version."
    return match.group("version")


def _expected_ghcr_identifier(payload: dict[str, Any]) -> str:
    name = payload.get("name")
    assert isinstance(name, str) and name, "server.json must define a non-empty public server name."
    match = re.match(r"^io\.github\.([^/]+)/([^/]+)$", name)
    assert match is not None, "server.json name must follow io.github.<owner>/<server>."
    owner, server_name = match.groups()
    version = payload.get("version")
    assert isinstance(version, str) and version
    return f"ghcr.io/{owner.lower()}/{server_name.lower()}:{version}"


def test_server_json_declares_public_name_and_semver_version() -> None:
    payload = _read_server_json()

    name = payload.get("name")
    assert isinstance(name, str) and name
    assert name.startswith("io.github.")
    assert "/" in name

    version = payload.get("version")
    assert isinstance(version, str) and version
    assert SEMVER_PATTERN.match(version), f"server.json version should use semantic versioning (got {version!r})."


def test_server_json_declares_at_least_one_oci_package() -> None:
    payload = _read_server_json()
    package_dicts = _iter_dicts(payload)

    has_oci_package = False
    for candidate in package_dicts:
        package_type = candidate.get("registryType") or candidate.get("registry_type") or candidate.get("type")
        if isinstance(package_type, str) and package_type.lower() == "oci":
            has_oci_package = True
            break

    assert has_oci_package, (
        "server.json must include at least one OCI package declaration (registryType/registry_type/type == 'oci')."
    )


def test_server_json_version_matches_pyproject_version() -> None:
    payload = _read_server_json()
    assert payload["version"] == _read_pyproject_version()


def test_server_json_primary_package_matches_public_server_name_ghcr_identifier() -> None:
    payload = _read_server_json()
    packages = payload.get("packages")
    assert isinstance(packages, list) and packages, "server.json must define at least one package entry."
    package = packages[0]
    assert isinstance(package, dict), "server.json package entries must be objects."
    assert package.get("registryType") == "oci"
    assert package.get("identifier") == _expected_ghcr_identifier(payload)
    transport = package.get("transport")
    assert isinstance(transport, dict), "server.json OCI package should declare transport metadata."
    assert transport.get("type") == "stdio"


def test_dockerfile_declares_mcp_server_label_matching_server_json() -> None:
    payload = _read_server_json()
    expected_name = payload["name"]
    text = DOCKERFILE.read_text(encoding="utf-8")

    label_match = re.search(
        r'io\.modelcontextprotocol\.server\.name\s*=\s*"?(?P<name>[^\s"\\]+)"?',
        text,
    )
    assert label_match is not None, "Dockerfile must set io.modelcontextprotocol.server.name label."
    assert label_match.group("name") == expected_name


def test_dockerfile_is_stdio_first_and_uses_explicit_entrypoint() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert re.search(r"^ENTRYPOINT\s+\[", text, re.MULTILINE), (
        "Dockerfile must use an explicit ENTRYPOINT for MCP package launches."
    )
    assert "paper_chaser_mcp.deployment_runner" not in text, (
        "Dockerfile default command should not force HTTP deployment runner mode."
    )
    assert "PAPER_CHASER_TRANSPORT=streamable-http" not in text, (
        "Dockerfile should not hardcode streamable-http as the default transport."
    )

    for default in re.findall(r"PAPER_CHASER_TRANSPORT=([^\s\\]+)", text):
        assert default.lower() in {"stdio", "${paper_chaser_transport:-stdio}"}
