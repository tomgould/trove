"""Source connectors for automated document acquisition."""

from core.sources.base import BaseSourceConnector, SyncResult
from core.sources.github_connector import GitHubSourceConnector
from core.sources.github_models import GitHubSourceConfig, RepoInfo
from core.sources.web_crawler import WebSourceConfig, WebSourceConnector

__all__ = [
    "BaseSourceConnector",
    "SyncResult",
    "GitHubSourceConnector",
    "GitHubSourceConfig",
    "RepoInfo",
    "WebSourceConfig",
    "WebSourceConnector",
]
