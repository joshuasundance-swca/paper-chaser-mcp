from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = REPO_ROOT / ".gitignore"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
README = REPO_ROOT / "README.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-azure.yml"
PUBLIC_MCP_PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-public-mcp-package.yml"
MCP_REGISTRY_PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-mcp-registry.yml"
GITHUB_RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-github-release.yml"
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


def _extract_readme_contents_entries(text: str) -> list[str]:
    entries: list[str] = []
    in_contents = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "## Contents":
            in_contents = True
            continue
        if not in_contents:
            continue
        if stripped.startswith("## "):
            break

        match = re.match(r"- \[(.+?)\]\(#.+\)$", stripped)
        if match is not None:
            entries.append(match.group(1))

    return entries


def _extract_readme_top_level_sections(text: str) -> list[str]:
    sections: list[str] = []

    for line in text.splitlines():
        if not line.startswith("## "):
            continue

        heading = line[3:].strip()
        if heading != "Contents":
            sections.append(heading)

    return sections


def test_gitignore_covers_local_env_and_planning_files() -> None:
    lines = _read_lines(GITIGNORE)

    assert REQUIRED_GITIGNORE_PATTERNS <= lines


def test_dockerignore_covers_local_env_and_planning_files() -> None:
    lines = _read_lines(DOCKERIGNORE)

    assert REQUIRED_DOCKERIGNORE_PATTERNS <= lines


def test_readme_contents_matches_top_level_sections() -> None:
    text = README.read_text(encoding="utf-8")

    assert _extract_readme_contents_entries(text) == _extract_readme_top_level_sections(text)


def test_dev_extras_include_shellcheck_for_local_workflow_lint_parity() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")

    assert '"shellcheck-py>=' in text


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


def test_public_mcp_publish_workflow_stays_ghcr_only() -> None:
    text = PUBLIC_MCP_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "mcp-publisher" not in text
    assert "Publish server to MCP Registry" not in text


def test_mcp_registry_publish_workflow_is_manual_only_and_pins_installer() -> None:
    text = MCP_REGISTRY_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "\n  push:\n" not in text
    assert "\n  pull_request:\n" not in text
    assert "packages: read" in text
    assert "id-token: write" in text
    assert "docker buildx imagetools inspect" in text
    assert "releases/latest/download" not in text
    assert "MCP_PUBLISHER_VERSION:" in text
    assert "sha256sum -c -" in text
    assert "./mcp-publisher login github-oidc" in text
    assert "./mcp-publisher publish" in text


def test_mcp_registry_publish_workflow_pins_actions_to_commit_shas() -> None:
    text = MCP_REGISTRY_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    for action in (
        "actions/checkout",
        "docker/setup-buildx-action",
        "docker/login-action",
    ):
        assert f"{action}@v" not in text
        assert action in text


def test_github_release_workflow_validates_prs_and_publishes_on_tags_or_manual() -> None:
    text = GITHUB_RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "workflow_dispatch:" in text
    assert "tags:" in text
    assert "- v*" in text
    assert "python -m build" in text
    assert "python -m twine check --strict dist/*" in text
    assert "sha256sum -- * > SHA256SUMS" in text
    assert "GH_REPO: ${{ github.repository }}" in text
    assert 'gh release view "$RELEASE_TAG" --repo "$GH_REPO"' in text
    assert 'gh release upload "$RELEASE_TAG" dist/* --repo "$GH_REPO" --clobber' in text
    assert "gh release create" in text
    assert "gh release upload" in text
    assert "--draft" in text
    assert "contents: write" in text
    assert "packages: write" not in text
    assert "id-token: write" not in text


def test_github_release_workflow_pins_actions_to_commit_shas() -> None:
    text = GITHUB_RELEASE_WORKFLOW.read_text(encoding="utf-8")

    for action in (
        "actions/checkout",
        "actions/setup-python",
        "actions/upload-artifact",
        "actions/download-artifact",
    ):
        assert f"{action}@v" not in text
        assert action in text


def test_pypi_publish_workflow_uses_oidc_environments_and_not_api_tokens() -> None:
    text = PYPI_PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in text
    assert "name: testpypi" in text
    assert "name: pypi" in text
    assert "vars.ENABLE_PYPI_PUBLISHING == 'true'" in text
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
