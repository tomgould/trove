"""
Agent template loader for Trove agent orchestration.

Loads YAML-defined reusable prompt configurations for specialized roles.
Agents are stored in the `agents/` directory at Trove root, each as a
separate YAML file with name, description, system_prompt, and tags.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class AgentLoader:
    """Loads and searches YAML agent templates.

    Args:
        agents_dir: Path to the agents directory.
    """

    def __init__(self, agents_dir: Path):
        self.agents_dir = Path(agents_dir)
        if not self.agents_dir.is_dir():
            logger.warning(f"Agents directory does not exist: {self.agents_dir}")
        self._cache: dict[str, dict] = {}
        self._loaded = False

    def _load_all(self):
        """Parse all YAML files in the agents directory into cache."""
        if self._loaded:
            return
        self._cache.clear()

        if not self.agents_dir.is_dir():
            self._loaded = True
            return

        for yaml_path in sorted(self.agents_dir.glob("*.yaml")):
            try:
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    logger.warning(f"Skipping malformed YAML: {yaml_path}")
                    continue
                slug = yaml_path.stem  # filename without .yaml
                data["_filename"] = slug
                data.setdefault("name", slug)
                data.setdefault("description", "")
                data.setdefault("system_prompt", "")
                data.setdefault("tags", [])
                self._cache[slug] = data
            except Exception as e:
                logger.warning(f"Failed to parse {yaml_path}: {e}")

        self._loaded = True

    def list(self, tag: str = None) -> list[dict]:
        """List all agents, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by (case-insensitive).

        Returns:
            List of agent summary dicts (name, description, tags, filename).
        """
        self._load_all()
        results = []
        for slug, data in self._cache.items():
            if tag:
                tags_lower = [t.lower() for t in data.get("tags", [])]
                if tag.lower() not in tags_lower:
                    continue
            results.append({
                "name": data["name"],
                "description": data["description"],
                "tags": data.get("tags", []),
                "filename": slug,
            })
        return results

    def get(self, agent_name: str) -> Optional[dict]:
        """Get full agent config by name or filename slug.

        Matching is case-insensitive on both the `name` field and the
        filename slug (without .yaml).

        Returns:
            Full agent dict including system_prompt, or None.
        """
        self._load_all()
        name_lower = agent_name.lower()

        # Try slug match first (fast)
        for slug, data in self._cache.items():
            if slug.lower() == name_lower:
                return dict(data)

        # Try name field match
        for slug, data in self._cache.items():
            if data["name"].lower() == name_lower:
                return dict(data)

        return None

    def search(self, query: str) -> list[dict]:
        """Search agents by substring match on name, description, and tags.

        Results are sorted by relevance: name match > description match > tag match.

        Returns:
            List of matching agent summary dicts.
        """
        self._load_all()
        query_lower = query.lower()
        scored: list[tuple[int, dict]] = []

        for slug, data in self._cache.items():
            score = 0
            name = data["name"].lower()
            desc = data["description"].lower()
            tags = [t.lower() for t in data.get("tags", [])]

            if query_lower in name:
                score += 3
            if query_lower in desc:
                score += 2
            if any(query_lower in t for t in tags):
                score += 1

            if score > 0:
                scored.append((score, {
                    "name": data["name"],
                    "description": data["description"],
                    "tags": data.get("tags", []),
                    "filename": slug,
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored]
