"""GitHub API client for source connectors.

Ported from github_auditer/src/auditer/github_client.py — trimmed to repo
discovery and metadata. Removes: org members, collaborators, file content,
events, commits, PRs, issues, repo settings, topics.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(GitHubAPIError):
    pass


class RateLimitError(GitHubAPIError):
    pass


def _mask_token(token: str | None) -> str:
    """Mask a token for safe logging."""
    if not token:
        return "(none)"
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


class GitHubClient:
    """Minimal GitHub API client for repo discovery.

    Supports: list repos for user/org, get single repo, pagination,
    rate limit handling, retries.
    """

    BASE_URL = "https://api.github.com"
    MAX_RETRIES = 3
    RETRY_BACKOFF = 1.0
    MAX_PAGES_ABSOLUTE = 1000

    def __init__(self, token: str | None = None):
        self._token = token
        self._authenticated_login: str | None = None
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as e:
                last_error = GitHubAPIError(f"Network error: {e}")
                time.sleep(self.RETRY_BACKOFF * (attempt + 1))
                continue

            if response.status_code == 401:
                token_hint = _mask_token(self._token)
                raise AuthenticationError(
                    f"Authentication failed (token: {token_hint}). "
                    f"Check your PAT has the required scopes.",
                    status_code=401,
                )

            if response.status_code == 403:
                remaining = response.headers.get("x-ratelimit-remaining")
                if remaining == "0":
                    reset_at = int(response.headers.get("x-ratelimit-reset", 0))
                    wait = max(0, reset_at - int(time.time())) + 1
                    raise RateLimitError(
                        f"Rate limit exceeded. Resets in {wait} seconds.",
                        status_code=403,
                    )
                raise GitHubAPIError(
                    f"Forbidden (403) for {url}", status_code=403
                )

            if response.status_code >= 500:
                last_error = GitHubAPIError(
                    f"Server error ({response.status_code}) for {url}",
                    status_code=response.status_code,
                )
                time.sleep(self.RETRY_BACKOFF * (attempt + 1))
                continue

            return response

        raise last_error or GitHubAPIError("Request failed after retries")

    def _get_paginated(
        self,
        url: str,
        params: dict | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        params = dict(params or {})
        params.setdefault("per_page", 100)
        page = 1

        while page <= self.MAX_PAGES_ABSOLUTE:
            params["page"] = page
            response = self._request("GET", url, params=params)

            if response.status_code == 404:
                raise GitHubAPIError(
                    f"Not found: {url} — check that the owner/org name is correct.",
                    status_code=404,
                )
            response.raise_for_status()

            items = response.json()
            if not items:
                break

            all_items.extend(items)

            link_header = response.headers.get("link", "")
            if 'rel="next"' not in link_header:
                break

            page += 1
            if max_pages is not None and page > max_pages:
                break

        return all_items

    def _get_authenticated_login(self) -> str | None:
        """Get the login of the authenticated user (cached)."""
        if self._authenticated_login is not None:
            return self._authenticated_login
        if not self._token:
            return None
        try:
            response = self._request("GET", "/user")
            response.raise_for_status()
            self._authenticated_login = response.json().get("login")
            return self._authenticated_login
        except AuthenticationError:
            raise
        except GitHubAPIError as e:
            logger.warning("Could not fetch authenticated user: %s", e)
            return None

    def is_org(self, owner: str) -> bool:
        """Check if an owner is a GitHub organization."""
        try:
            response = self._request("GET", f"/orgs/{owner}")
            return response.status_code == 200
        except AuthenticationError:
            raise
        except GitHubAPIError as e:
            logger.warning("Could not determine account type for %s: %s", owner, e)
            return False

    def list_repos(
        self,
        owner: str,
        account_type: str = "user",
        include_private: bool = True,
        include_forks: bool = False,
    ) -> list[dict[str, Any]]:
        """List repos for a user or org, with basic filtering."""
        if account_type == "org":
            url = f"/orgs/{owner}/repos"
            params = {"sort": "pushed", "type": "all"}
        elif self._token and self._get_authenticated_login():
            login = self._authenticated_login
            if login and login.lower() == owner.lower():
                url = "/user/repos"
                params = {"sort": "pushed", "affiliation": "owner"}
            else:
                url = f"/users/{owner}/repos"
                params = {"sort": "pushed", "type": "owner"}
        else:
            url = f"/users/{owner}/repos"
            params = {"sort": "pushed", "type": "owner"}

        repos = self._get_paginated(url, params)

        # Apply visibility filters
        if not include_private:
            repos = [r for r in repos if not r.get("private", False)]
        if not include_forks:
            repos = [r for r in repos if not r.get("fork", False)]

        return repos

    def get_repo(self, owner: str, repo: str) -> dict[str, Any] | None:
        """Fetch a single repo's metadata. Returns None on 404."""
        response = self._request("GET", f"/repos/{owner}/{repo}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
