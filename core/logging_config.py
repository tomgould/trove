"""
Client-configurable audit logging.

Provides configuration for audit trail behavior per client,
allowing customization of what gets logged and how.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AuditConfig:
    """
    Client-level audit configuration.

    Loaded from client.yaml 'audit' section with sensible defaults.

    Example client.yaml:
        audit:
          enabled: true
          log_level: "full"
          include_chunks: true
          include_scores: true
          include_timing: true
          retention_days: 90
    """

    enabled: bool = True
    log_level: str = "full"  # "full", "minimal", "none"
    include_chunks: bool = True
    include_scores: bool = True
    include_timing: bool = True
    retention_days: Optional[int] = None

    def should_log(self) -> bool:
        """Check if logging is enabled and level is not 'none'."""
        return self.enabled and self.log_level != "none"

    def is_minimal(self) -> bool:
        """Check if minimal logging is configured."""
        return self.log_level == "minimal"

    def is_full(self) -> bool:
        """Check if full logging is configured."""
        return self.log_level == "full"


def load_audit_config(client_config: dict) -> AuditConfig:
    """
    Load audit configuration from client.yaml.

    Falls back to defaults if audit section is missing.

    Args:
        client_config: Full client configuration dict

    Returns:
        AuditConfig instance
    """
    audit_section = client_config.get("audit", {})

    return AuditConfig(
        enabled=audit_section.get("enabled", True),
        log_level=audit_section.get("log_level", "full"),
        include_chunks=audit_section.get("include_chunks", True),
        include_scores=audit_section.get("include_scores", True),
        include_timing=audit_section.get("include_timing", True),
        retention_days=audit_section.get("retention_days"),
    )


def get_default_audit_config() -> AuditConfig:
    """Get default audit configuration."""
    return AuditConfig()
