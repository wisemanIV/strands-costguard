"""Configuration classes for Cost Guard."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class FailureMode(str, Enum):
    """How to handle failures in policy store or metrics export."""

    FAIL_OPEN = "fail_open"  # Allow operations to proceed
    FAIL_CLOSED = "fail_closed"  # Block operations


class PolicySource(Protocol):
    """Protocol for policy sources."""

    def load_budgets(self) -> list[dict]:
        """Load budget specifications."""
        ...

    def load_routing_policies(self) -> list[dict]:
        """Load routing policies."""
        ...

    def load_pricing(self) -> dict:
        """Load pricing table."""
        ...


@dataclass
class CostGuardConfig:
    """Main configuration for Cost Guard."""

    policy_source: PolicySource
    failure_mode: FailureMode = FailureMode.FAIL_OPEN
    policy_refresh_interval_seconds: int = 300
    enable_budget_enforcement: bool = True
    enable_routing: bool = True
    enable_metrics: bool = True
    include_run_id_in_metrics: bool = False  # High cardinality warning
    currency: str = "USD"
    default_max_iterations_per_run: int = 50
    default_max_tool_calls_per_run: int = 100
    default_max_tokens_per_run: int = 100000
    log_level: str = "INFO"
