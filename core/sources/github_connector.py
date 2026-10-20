"""GitHub source connector for Trove.

Discovers repos from GitHub accounts/orgs, clones them to a managed directory,
keeps them synced via git fetch, and reports changed files for incremental
reindexing.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from core.sources.base import BaseSourceConnector, SyncResult
from core.sources.git_ops import CloneResult, clone_repo, fetch_repo, get_default_branch
from core.sources.github_client import AuthenticationError, GitHubAPIError, GitHubClient
from core.sources.github_models import GitHubSourceConfig, RepoInfo

logger = logging.getLogger(__name__)


class GitHubSourceConnector(BaseSourceConnector):
    """Sync GitHub repos to a local directory for Trove indexing."""

    def __init__(self, config: GitHubSourceConfig):
        self.config = config

    def sync(self, target_dir: str) -> SyncResult:
        """Discover and sync repos to target_dir/github/{owner}/{repo}/."""
        result = SyncResult(source_type="github")
        target = Path(target_dir)
        github_dir = target / "github"
        github_dir.mkdir(parents=True, exist_ok=True)

        # Discover repos
        try:
            repos = self._discover_repos()
        except GitHubAPIError as e:
            result.errors.append(f"GitHub API error: {e}")
            return result

        if not repos:
            logger.info("No repos found for %s", self.config.owner)
            return result

        logger.info("Found %d repos for %s", len(repos), self.config.owner)

        # Sync each repo
        for repo in repos:
            repo_dir = github_dir / repo.owner / repo.name
            try:
                clone_result = self._sync_repo(repo, repo_dir)
                if not clone_result.success:
                    result.errors.append(
                        f"{repo.owner}/{repo.name}: {clone_result.message}"
                    )
                    continue

                if clone_result.changed_files:
                    result.files_updated += len(clone_result.changed_files)
                elif clone_result.is_fresh_clone:
                    result.files_added += 1  # count as 1 new repo

            except Exception as e:
                result.errors.append(f"{repo.owner}/{repo.name}: {e}")
                logger.exception("Error syncing %s/%s", repo.owner, repo.name)

        # Save sync metadata
        try:
            self._save_sync_metadata(github_dir, repos)
        except OSError as e:
            result.errors.append(f"Failed to save sync metadata: {e}")
            logger.warning("Could not save sync metadata: %s", e)

        return result

    def status(self) -> dict:
        """Return current sync status."""
        return {
            "source_type": "github",
            "owner": self.config.owner,
            "repo": self.config.repo,
            "has_token": self.config.token is not None,
            "sync_on_ingest": self.config.sync_on_ingest,
        }

    def _discover_repos(self) -> list[RepoInfo]:
        """Use GitHub API to discover repos based on config."""
        token = self.config.token

        # Single repo mode
        if self.config.repo and self.config.owner:
            client = GitHubClient(token=token)
            try:
                data = client.get_repo(self.config.owner, self.config.repo)
                if data is None:
                    logger.warning(
                        "Repo %s/%s not found",
                        self.config.owner,
                        self.config.repo,
                    )
                    return []
                return [RepoInfo.from_api_response(data)]
            finally:
                client.close()

        # Org/user mode
        if not self.config.owner:
            logger.warning("No owner configured for GitHub source connector")
            return []

        # Detect account type and list repos using a single client
        with GitHubClient(token=token) as client:
            account_type = "org" if client.is_org(self.config.owner) else "user"
            raw_repos = client.list_repos(
                self.config.owner,
                account_type=account_type,
                include_private=self.config.include_private,
                include_forks=self.config.include_forks,
            )

        repos = [RepoInfo.from_api_response(r) for r in raw_repos]

        # Apply include/exclude filters
        repos = self._filter_repos(repos)

        return repos

    def _filter_repos(self, repos: list[RepoInfo]) -> list[RepoInfo]:
        """Apply include_repos and exclude_repos filters."""
        if self.config.include_repos:
            included = {name.lower() for name in self.config.include_repos}
            repos = [r for r in repos if r.name.lower() in included]

        if self.config.exclude_repos:
            excluded = {name.lower() for name in self.config.exclude_repos}
            repos = [r for r in repos if r.name.lower() not in excluded]

        return repos

    def _sync_repo(self, repo: RepoInfo, repo_dir: Path) -> CloneResult:
        """Clone or fetch a single repo."""
        token = self.config.token

        if repo_dir.exists() and (repo_dir / ".git").is_dir():
            # Existing repo — fetch updates
            logger.info("Fetching %s/%s", repo.owner, repo.name)
            return fetch_repo(
                repo_dir,
                repo.clone_url,
                default_branch=repo.default_branch,
                token=token,
            )
        else:
            # New repo — clone
            logger.info("Cloning %s/%s", repo.owner, repo.name)
            return clone_repo(repo.clone_url, repo_dir, token=token)

    def _save_sync_metadata(
        self, github_dir: Path, repos: list[RepoInfo]
    ) -> None:
        """Save sync metadata for status reporting."""
        metadata = {
            "last_sync": time.time(),
            "owner": self.config.owner,
            "repos": [
                {
                    "name": r.name,
                    "owner": r.owner,
                    "default_branch": r.default_branch,
                    "private": r.private,
                }
                for r in repos
            ],
        }
        metadata_path = github_dir / self.config.owner / ".sync_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))
