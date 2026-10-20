"""
Knowledge graph data models.

Entity and Relationship dataclasses with merge/dedup logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    """A named entity extracted from document chunks."""

    name: str
    entity_type: str
    description: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique key for dedup: 'normalized_name::type'."""
        return f"{self.name.lower().strip()}::{self.entity_type.lower().strip()}"

    def merge(self, other: Entity) -> None:
        """Merge another entity into this one, accumulating sources and aliases."""
        for cid in other.source_chunk_ids:
            if cid not in self.source_chunk_ids:
                self.source_chunk_ids.append(cid)
        for alias in other.aliases:
            if alias not in self.aliases:
                self.aliases.append(alias)
        # Keep the longer description
        if len(other.description) > len(self.description):
            self.description = other.description

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_chunk_ids": self.source_chunk_ids,
            "aliases": self.aliases,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Entity:
        return cls(
            name=data["name"],
            entity_type=data["entity_type"],
            description=data.get("description", ""),
            source_chunk_ids=data.get("source_chunk_ids", []),
            aliases=data.get("aliases", []),
        )


@dataclass
class Relationship:
    """A relationship between two entities."""

    source_entity: str  # Entity key
    target_entity: str  # Entity key
    relationship_type: str
    description: str = ""
    weight: float = 1.0  # Confidence 0-1
    source_chunk_ids: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique key: 'source--type--target'."""
        return f"{self.source_entity}--{self.relationship_type.lower().strip()}--{self.target_entity}"

    def merge(self, other: Relationship) -> None:
        """Merge another relationship, taking max weight and accumulating sources."""
        self.weight = max(self.weight, other.weight)
        for cid in other.source_chunk_ids:
            if cid not in self.source_chunk_ids:
                self.source_chunk_ids.append(cid)
        if len(other.description) > len(self.description):
            self.description = other.description

    def to_dict(self) -> dict:
        return {
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "weight": self.weight,
            "source_chunk_ids": self.source_chunk_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Relationship:
        return cls(
            source_entity=data["source_entity"],
            target_entity=data["target_entity"],
            relationship_type=data["relationship_type"],
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            source_chunk_ids=data.get("source_chunk_ids", []),
        )
