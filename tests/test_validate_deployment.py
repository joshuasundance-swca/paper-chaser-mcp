from __future__ import annotations

import pytest

from scripts import validate_deployment


def test_expected_ghcr_identifier_accepts_canonical_https_github_repo_url() -> None:
    assert (
        validate_deployment._expected_ghcr_identifier(
            "https://github.com/Owner/Repo",
            "1.2.3",
        )
        == "ghcr.io/owner/repo:1.2.3"
    )


@pytest.mark.parametrize(
    "repository_url",
    [
        "http://github.com/owner/repo",
        "https://www.github.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/owner/repo/tree/main",
    ],
)
def test_expected_ghcr_identifier_rejects_non_canonical_repository_urls(
    repository_url: str,
) -> None:
    with pytest.raises(
        SystemExit,
        match=(
            "server.json repository.url must point to an "
            "https://github.com/<owner>/<repo> path"
        ),
    ):
        validate_deployment._expected_ghcr_identifier(repository_url, "1.2.3")
