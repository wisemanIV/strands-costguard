"""Configuration classes for Cost Guard."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from strands_costguard.persistence.valkey_store import ValkeyBudgetStore


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
    """Main configuration for Cost Guard.

    Args:
        policy_source: Source for loading policies (file or env)
        budget_store: Optional Valkey store for persistent budget state.
            When provided, budget state survives application restarts.

            Example:
                import valkey
                from strands_costguard.persistence import ValkeyBudgetStore

                client = valkey.Valkey(host="localhost", port=6379)
                store = ValkeyBudgetStore(client)

                config = CostGuardConfig(
                    policy_source=FilePolicySource(path="./policies"),
                    budget_store=store,
                )
    """

    policy_source: PolicySource
    budget_store: Optional["ValkeyBudgetStore"] = None
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
