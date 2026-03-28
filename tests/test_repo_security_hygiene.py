from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = REPO_ROOT / ".gitignore"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-azure.yml"
PUBLIC_MCP_PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-public-mcp-package.yml"
PYPI_PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml"

REQUIRED_GITIGNORE_PATTERNS = {
    ".env",
    ".env.local",
    ".env.*",
    "!.env.example",
    ".local-*",
    ".local-rollout-artifacts/",
}

REQUIRED_DOCKERIGNORE_PATTERNS = {
    ".env",
    ".env.local",
    ".env.*",
    "!.env.example",
    ".local-*",
}

DEPLOYMENT_IDENTIFIER_SECRET_KEYS = {
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "ACR_NAME",
    "IMAGE_REPOSITORY",
}


def _read_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_gitignore_covers_local_env_and_planning_files() -> None:
    lines = _read_lines(GITIGNORE)

    assert REQUIRED_GITIGNORE_PATTERNS <= lines


def test_dockerignore_covers_local_env_and_planning_files() -> None:
    lines = _read_lines(DOCKERIGNORE)

    assert REQUIRED_DOCKERIGNORE_PATTERNS <= lines


def test_deploy_workflow_uses_secrets_for_deployment_identifiers() -> None:
    text = DEPLOY_WORKFLOW.read_text(encoding="utf-8")

    for key in DEPLOYMENT_IDENTIFIER_SECRET_KEYS:
        assert f"secrets.{key}" in text
        assert f"vars.{key}" not in text


def test_deploy_workflow_keeps_only_runner_labels_as_variable_override() -> None:
    text = DEPLOY_WORKFLOW.read_text(encoding="utf-8")

    assert "vars.AZURE_PRIVATE_RUNNER_LABELS_JSON" in text


def test_public_mcp_publish_workflow_pins_docker_actions_to_commit_shas() -> None:
    text = PUBLIC_MCP_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    for action in (
        "docker/setup-buildx-action",
        "docker/setup-qemu-action",
        "docker/build-push-action",
        "docker/login-action",
        "docker/metadata-action",
    ):
        assert f"{action}@v" not in text
        assert action in text


def test_public_mcp_publish_workflow_pins_mcp_publisher_install() -> None:
    text = PUBLIC_MCP_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "releases/latest/download" not in text
    assert "MCP_PUBLISHER_VERSION:" in text
    assert "sha256sum -c -" in text


def test_pypi_publish_workflow_uses_oidc_environments_and_not_api_tokens() -> None:
    text = PYPI_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in text
    assert "name: testpypi" in text
    assert "name: pypi" in text
    assert "repository-url: https://test.pypi.org/legacy/" in text
    assert "secrets.PYPI_API_TOKEN" not in text
    assert "secrets.TEST_PYPI_API_TOKEN" not in text
    assert "pypa/gh-action-pypi-publish@" in text


def test_pypi_publish_workflow_pins_github_actions_to_commit_shas() -> None:
    text = PYPI_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    for action in (
        "actions/checkout",
        "actions/setup-python",
        "actions/upload-artifact",
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
    ):
        assert f"{action}@v" not in text
        assert action in text
