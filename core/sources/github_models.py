"""Data models for GitHub source connector.

Ported from github_auditer — trimmed to what Trove needs for repo discovery and cloning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class GitHubSourceConfig:
    """Configuration for a GitHub source in client.yaml."""

    owner: str = ""
    repo: str = ""  # single repo shorthand: "owner/repo"
    token_env: str = ""  # env var name for GitHub token
    include_repos: list[str] = field(default_factory=list)
    exclude_repos: list[str] = field(default_factory=list)
    include_forks: bool = False
    include_private: bool = True
    sync_on_ingest: bool = True

    def __post_init__(self):
        # Parse "owner/repo" shorthand
        if self.repo and "/" in self.repo and not self.owner:
            self.owner, self.repo = self.repo.split("/", 1)

    @property
    def token(self) -> str | None:
        """Read token from environment variable."""
        if not self.token_env:
            return None
        return os.environ.get(self.token_env)

    @classmethod
    def from_dict(cls, data: dict) -> GitHubSourceConfig:
        return cls(
            owner=data.get("owner", ""),
            repo=data.get("repo", ""),
            token_env=data.get("token_env", ""),
            include_repos=data.get("include_repos", []),
            exclude_repos=data.get("exclude_repos", []),
            include_forks=data.get("include_forks", False),
            include_private=data.get("include_private", True),
            sync_on_ingest=data.get("sync_on_ingest", True),
        )


@dataclass
class RepoInfo:
    """Metadata about a GitHub repository."""

    name: str
    owner: str
    url: str
    clone_url: str
    private: bool
    description: str
    default_branch: str
    size_kb: int
    is_fork: bool

    @classmethod
    def from_api_response(cls, data: dict) -> RepoInfo:
        return cls(
            name=data["name"],
            owner=data["owner"]["login"],
            url=data.get("html_url", ""),
            clone_url=data.get("clone_url", ""),
            private=data.get("private", False),
            description=data.get("description") or "",
            default_branch=data.get("default_branch", "main"),
            size_kb=data.get("size", 0),
            is_fork=data.get("fork", False),
        )
