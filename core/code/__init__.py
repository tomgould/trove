"""
Code indexing module for Trove.

Provides tree-sitter based parsing and chunking for source code files.
Supports Python, PHP, JavaScript, TypeScript, HTML, CSS, and YAML.
"""

from .parser import CodeParser
from .chunker import CodeChunker
from .git_utils import detect_repos, RepoInfo
from .incremental import CodeIndexState, compute_chunk_id

__all__ = [
    "CodeParser",
    "CodeChunker",
    "detect_repos",
    "RepoInfo",
    "CodeIndexState",
    "compute_chunk_id",
]
