"""
Reference Configuration for RAG.

Provides configurable reference patterns for different document domains
(legal, technical, academic, compliance, etc.).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# Built-in reference patterns by domain
DOMAIN_PATTERNS = {
    # Compliance/Regulatory domain (default)
    "compliance": {
        "appendix": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?(?:Appendix|Annex|Annexure)\s+([A-Z0-9]+(?:\.[0-9]+)?)\b',
                r'\b(?:refer\s+to\s+)?(?:Appendix|Annex|Annexure)\s+([A-Z0-9]+(?:\.[0-9]+)?)\b',
                r'\bas\s+(?:set\s+out|listed|detailed|described)\s+in\s+(?:Appendix|Annex)\s+([A-Z0-9]+)\b',
            ],
            "query_templates": ["Appendix {id}", "Annex {id}"],
        },
        "section": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?Section\s+(\d+(?:\.\d+)*)\b',
                r'\b(?:refer\s+to\s+)?Section\s+(\d+(?:\.\d+)*)\b',
            ],
            "query_templates": ["Section {id}"],
        },
        "schedule": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?Schedule\s+(\d+|[A-Z])\b',
                r'\b(?:refer\s+to\s+)?Schedule\s+(\d+|[A-Z])\b',
            ],
            "query_templates": ["Schedule {id}"],
        },
        "table": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?Table\s+(\d+(?:\.\d+)?|[A-Z])\b',
            ],
            "query_templates": ["Table {id}"],
        },
    },

    # Legal domain (contracts, case law)
    "legal": {
        "exhibit": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?Exhibit\s+([A-Z0-9]+)\b',
                r'\b(?:refer\s+to\s+)?Exhibit\s+([A-Z0-9]+)\b',
                r'\b(?:attached\s+)?(?:as\s+)?Exhibit\s+([A-Z0-9]+)\b',
            ],
            "query_templates": ["Exhibit {id}"],
        },
        "clause": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?Clause\s+(\d+(?:\.\d+)*)\b',
                r'\b(?:pursuant\s+to\s+)?Clause\s+(\d+(?:\.\d+)*)\b',
            ],
            "query_templates": ["Clause {id}"],
        },
        "article": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?Article\s+(\d+|[IVXLC]+)\b',
                r'\b(?:under\s+)?Article\s+(\d+|[IVXLC]+)\b',
            ],
            "query_templates": ["Article {id}"],
        },
        "recital": {
            "priority": "low",
            "patterns": [
                r'\b(?:see\s+)?Recital\s+([A-Z]|\d+)\b',
            ],
            "query_templates": ["Recital {id}"],
        },
        "appendix": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?(?:Appendix|Annex)\s+([A-Z0-9]+)\b',
            ],
            "query_templates": ["Appendix {id}", "Annex {id}"],
        },
        "schedule": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?Schedule\s+(\d+|[A-Z])\b',
            ],
            "query_templates": ["Schedule {id}"],
        },
    },

    # Technical domain (APIs, RFCs, issues)
    "technical": {
        "rfc": {
            "priority": "high",
            "patterns": [
                r'\bRFC\s*(\d+)\b',
                r'\b(?:see\s+)?RFC\s*(\d+)\b',
            ],
            "query_templates": ["RFC {id}"],
        },
        "issue": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?(?:Issue|Bug)\s*#?(\d+)\b',
                r'\b(?:fixes|closes|resolves)\s+#(\d+)\b',
            ],
            "query_templates": ["Issue {id}", "Bug {id}"],
        },
        "pr": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?(?:PR|Pull Request)\s*#?(\d+)\b',
            ],
            "query_templates": ["PR {id}", "Pull Request {id}"],
        },
        "section": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?Section\s+(\d+(?:\.\d+)*)\b',
            ],
            "query_templates": ["Section {id}"],
        },
    },

    # Academic domain (papers, textbooks)
    "academic": {
        "equation": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?(?:Equation|Eq\.?)\s*\(?(\d+(?:\.\d+)?)\)?\b',
            ],
            "query_templates": ["Equation {id}"],
        },
        "theorem": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?Theorem\s+(\d+(?:\.\d+)?)\b',
            ],
            "query_templates": ["Theorem {id}"],
        },
        "lemma": {
            "priority": "high",
            "patterns": [
                r'\b(?:see\s+)?Lemma\s+(\d+(?:\.\d+)?)\b',
            ],
            "query_templates": ["Lemma {id}"],
        },
        "definition": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?Definition\s+(\d+(?:\.\d+)?)\b',
            ],
            "query_templates": ["Definition {id}"],
        },
        "figure": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)?)\b',
            ],
            "query_templates": ["Figure {id}"],
        },
        "table": {
            "priority": "medium",
            "patterns": [
                r'\b(?:see\s+)?Table\s+(\d+(?:\.\d+)?)\b',
            ],
            "query_templates": ["Table {id}"],
        },
        "chapter": {
            "priority": "low",
            "patterns": [
                r'\b(?:see\s+)?Chapter\s+(\d+)\b',
            ],
            "query_templates": ["Chapter {id}"],
        },
        "section": {
            "priority": "low",
            "patterns": [
                r'\b(?:see\s+)?Section\s+(\d+(?:\.\d+)*)\b',
            ],
            "query_templates": ["Section {id}"],
        },
    },
}

# Default semantic patterns (phrases to search for in context)
DEFAULT_SEMANTIC_PATTERNS = {
    "compound_phrases": [
        "high risk countries",
        "higher risk countries",
        "due diligence",
        "material adverse change",
        "representations and warranties",
    ],
}


@dataclass
class ReferencePattern:
    """Configuration for a single reference type."""
    name: str
    priority: str = "medium"  # "high", "medium", "low"
    patterns: list[str] = field(default_factory=list)
    query_templates: list[str] = field(default_factory=list)
    _compiled: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Compile regex patterns."""
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def find_matches(self, text: str) -> list[dict]:
        """Find all matches of this pattern in text."""
        matches = []
        seen = set()

        for regex in self._compiled:
            for match in regex.finditer(text):
                ref_id = match.group(1).upper()
                key = f"{self.name}:{ref_id}"

                if key not in seen:
                    seen.add(key)
                    matches.append({
                        "type": self.name,
                        "id": ref_id,
                        "match": match.group(0),
                        "key": key,
                        "priority": self.priority,
                    })

        return matches

    def build_queries(self, ref_id: str) -> list[str]:
        """Build search queries for a reference ID."""
        return [tmpl.format(id=ref_id) for tmpl in self.query_templates]


@dataclass
class ReferenceConfig:
    """
    Configuration for reference detection and resolution.

    Can be loaded from client.yaml or created programmatically.
    """
    enabled: bool = True
    mode: str = "extend"  # "extend" (add to defaults) or "replace" (only use custom)
    domain: str = "compliance"  # Base domain for built-in patterns
    patterns: dict = field(default_factory=dict)  # Custom pattern definitions
    semantic_patterns: dict = field(default_factory=dict)  # Semantic phrase patterns

    _compiled_patterns: list[ReferencePattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Build compiled patterns from config."""
        self._build_patterns()

    def _build_patterns(self):
        """Build ReferencePattern objects from config."""
        self._compiled_patterns = []

        # Start with domain defaults if mode is "extend"
        if self.mode == "extend" and self.domain in DOMAIN_PATTERNS:
            for name, config in DOMAIN_PATTERNS[self.domain].items():
                self._compiled_patterns.append(ReferencePattern(
                    name=name,
                    priority=config.get("priority", "medium"),
                    patterns=config.get("patterns", []),
                    query_templates=config.get("query_templates", []),
                ))

        # Add/override with custom patterns
        for name, config in self.patterns.items():
            # Check if we're overriding an existing pattern
            existing = next((p for p in self._compiled_patterns if p.name == name), None)
            if existing:
                # Replace existing
                self._compiled_patterns.remove(existing)

            self._compiled_patterns.append(ReferencePattern(
                name=name,
                priority=config.get("priority", "medium"),
                patterns=config.get("patterns", config.get("regex", [])),
                query_templates=config.get("query_templates", []),
            ))

    def get_patterns(self) -> list[ReferencePattern]:
        """Get all compiled patterns."""
        return self._compiled_patterns

    def get_semantic_phrases(self) -> list[str]:
        """Get semantic phrases for context extraction."""
        phrases = []

        # Add default phrases
        phrases.extend(DEFAULT_SEMANTIC_PATTERNS.get("compound_phrases", []))

        # Add custom phrases
        custom = self.semantic_patterns.get("compound_phrases", [])
        phrases.extend(custom)

        return list(set(phrases))

    def detect_references(self, text: str) -> list[dict]:
        """
        Detect all references in text using configured patterns.

        Args:
            text: Text to scan for references

        Returns:
            List of detected references sorted by priority
        """
        all_refs = []

        for pattern in self._compiled_patterns:
            refs = pattern.find_matches(text)
            all_refs.extend(refs)

        # Sort by priority (high first) then by position in text
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_refs.sort(key=lambda r: priority_order.get(r["priority"], 1))

        return all_refs

    def build_queries_for_reference(self, ref: dict) -> list[str]:
        """Build search queries for a detected reference."""
        pattern = next((p for p in self._compiled_patterns if p.name == ref["type"]), None)
        if pattern:
            return pattern.build_queries(ref["id"])
        return [f"{ref['type'].title()} {ref['id']}"]


def load_reference_config(config_dict: dict) -> ReferenceConfig:
    """
    Load ReferenceConfig from a dictionary (e.g., from client.yaml).

    Args:
        config_dict: Dictionary with reference configuration
                     {"enabled": bool, "mode": str, "domain": str, "patterns": {...}}

    Returns:
        ReferenceConfig instance
    """
    if not config_dict:
        return ReferenceConfig()

    return ReferenceConfig(
        enabled=config_dict.get("enabled", True),
        mode=config_dict.get("mode", "extend"),
        domain=config_dict.get("domain", "compliance"),
        patterns=config_dict.get("patterns", {}),
        semantic_patterns=config_dict.get("semantic_patterns", {}),
    )


def get_default_config(domain: str = "compliance") -> ReferenceConfig:
    """
    Get a default ReferenceConfig for a domain.

    Args:
        domain: Domain name ("compliance", "legal", "technical", "academic")

    Returns:
        ReferenceConfig with domain defaults
    """
    return ReferenceConfig(domain=domain)
