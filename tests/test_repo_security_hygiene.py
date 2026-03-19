from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = REPO_ROOT / ".gitignore"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-azure.yml"

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
