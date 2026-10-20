"""Git operations for source connectors.

Ported from github_auditer/src/auditer/git_ops.py — trimmed to clone, fetch, and
changed-file detection. Security utilities inlined.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# GitHub token patterns for sanitisation
_TOKEN_PATTERN = re.compile(
    r"(ghp_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{80,}|"
    r"gho_[A-Za-z0-9]{36,}|ghs_[A-Za-z0-9]{36,}|"
    r"ghu_[A-Za-z0-9]{36,}|ghr_[A-Za-z0-9]{36,})"
)
_URL_TOKEN_PATTERN = re.compile(r"(https?://)([^@:]+:[^@]+)@(github\.com)")


def _sanitise_output(text: str) -> str:
    """Remove tokens from git command output."""
    result = _TOKEN_PATTERN.sub("****", text)
    return _URL_TOKEN_PATTERN.sub(r"\1****@\3", result)


class GitError(Exception):
    pass


def _validate_name(name: str) -> None:
    if not _SAFE_NAME_RE.match(name):
        raise GitError(f"Invalid repository name: {name!r}")
    if ".." in name:
        raise GitError(f"Path traversal detected in name: {name!r}")


def _make_askpass_script(token: str) -> str:
    """Create a temporary GIT_ASKPASS script that echoes the token."""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="git_askpass_")
    try:
        safe_token = token.replace("'", "'\\''")
        os.write(fd, f"#!/bin/sh\necho '{safe_token}'\n".encode())
    finally:
        os.close(fd)
    os.chmod(path, 0o700)
    return path


def _run_git(
    args: list[str],
    token: str | None = None,
    cwd: str | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a git command with optional token authentication."""
    env = os.environ.copy()
    askpass_script = None

    if token:
        askpass_script = _make_askpass_script(token)
        env["GIT_ASKPASS"] = askpass_script
        env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Git command timed out after %ds: git %s", timeout, " ".join(args))
        result = subprocess.CompletedProcess(
            args=["git"] + args,
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
        )
    finally:
        if askpass_script:
            try:
                os.unlink(askpass_script)
            except FileNotFoundError:
                pass  # Already cleaned up
            except OSError as e:
                logger.warning(
                    "Failed to clean up askpass script at %s: %s — "
                    "this file contains a token and should be manually removed",
                    askpass_script, e,
                )

    return result


@dataclass
class CloneResult:
    """Result of a clone or fetch operation."""

    success: bool
    message: str
    changed_files: list[str] = field(default_factory=list)
    is_fresh_clone: bool = False


def clone_repo(
    clone_url: str,
    target_dir: Path,
    token: str | None = None,
) -> CloneResult:
    """Clone a repository to target_dir."""
    _validate_name(target_dir.name)

    if token:
        auth_url = clone_url.replace("https://", "https://oauth2@")
    else:
        auth_url = clone_url

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    result = _run_git(["clone", auth_url, str(target_dir)], token=token)

    if result.returncode != 0:
        msg = _sanitise_output(result.stderr)
        return CloneResult(success=False, message=f"Clone failed: {msg}")

    # Clean remote URL (remove auth)
    cleanup = _run_git(["config", "remote.origin.url", clone_url], cwd=str(target_dir))
    if cleanup.returncode != 0:
        logger.warning("Failed to clean remote URL after clone in %s", target_dir)

    return CloneResult(success=True, message="Cloned successfully", is_fresh_clone=True)


def fetch_repo(
    repo_dir: Path,
    clone_url: str,
    default_branch: str = "main",
    token: str | None = None,
) -> CloneResult:
    """Fetch and reset a repository to match remote. Returns list of changed files."""
    if not (repo_dir / ".git").is_dir():
        return CloneResult(success=False, message=f"Not a git repository: {repo_dir}")

    cwd = str(repo_dir)

    # Get current HEAD before fetch
    old_head = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    old_hash = old_head.stdout.strip() if old_head.returncode == 0 else ""

    # Set authenticated remote temporarily
    if token:
        auth_url = clone_url.replace("https://", "https://oauth2@")
        _run_git(["config", "remote.origin.url", auth_url], cwd=cwd)

    try:
        result = _run_git(["fetch", "--all", "--prune"], token=token, cwd=cwd)
        if result.returncode != 0:
            return CloneResult(
                success=False,
                message=f"Fetch failed: {_sanitise_output(result.stderr)}",
            )

        # Check remote branch exists
        check = _run_git(
            ["rev-parse", "--verify", f"origin/{default_branch}"], cwd=cwd
        )
        if check.returncode != 0:
            return CloneResult(success=True, message="Repository appears empty")

        # Reset to remote
        result = _run_git(
            ["reset", "--hard", f"origin/{default_branch}"], cwd=cwd
        )
        if result.returncode != 0:
            return CloneResult(
                success=False,
                message=f"Reset failed: {_sanitise_output(result.stderr)}",
            )

        # Detect changed files
        changed_files: list[str] = []
        if old_hash:
            new_head = _run_git(["rev-parse", "HEAD"], cwd=cwd)
            new_hash = new_head.stdout.strip() if new_head.returncode == 0 else ""

            if old_hash != new_hash and new_hash:
                diff = _run_git(
                    ["diff", "--name-only", old_hash, new_hash], cwd=cwd
                )
                if diff.returncode == 0 and diff.stdout.strip():
                    changed_files = [
                        f for f in diff.stdout.strip().splitlines() if f.strip()
                    ]

        if changed_files:
            msg = f"Updated: {len(changed_files)} files changed"
        else:
            msg = "Already up to date"

        return CloneResult(
            success=True, message=msg, changed_files=changed_files
        )

    finally:
        # Always reset remote URL back to clean
        cleanup = _run_git(["config", "remote.origin.url", clone_url], cwd=cwd)
        if cleanup.returncode != 0:
            logger.warning("Failed to clean remote URL after fetch in %s", repo_dir)


def get_default_branch(repo_dir: Path) -> str:
    """Get the default branch name from a local repo."""
    result = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo_dir))
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    logger.warning("Could not detect default branch for %s, falling back to 'main'", repo_dir)
    return "main"
