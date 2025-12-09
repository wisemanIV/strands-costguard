"""Configuration classes for Cost Guard."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


class FailureMode(str, Enum):
    """How to handle failures in policy store or metrics export."""

    FAIL_OPEN = "fail_open"  # Allow operations to proceed
    FAIL_CLOSED = "fail_closed"  # Block operations


@dataclass
class OtelConfig:
    """Configuration for OpenTelemetry metrics export."""

    enabled: bool = True
    endpoint: str = "http://localhost:4317"
    use_grpc: bool = True
    service_name: str = "strands-service"
    service_namespace: Optional[str] = None
    deployment_environment: str = "development"
    export_interval_ms: int = 60000
    export_timeout_ms: int = 30000
    max_export_batch_size: int = 512
    include_run_id_attribute: bool = False  # High cardinality warning
    resource_attributes: dict[str, str] = field(default_factory=dict)

    def get_resource_attributes(self) -> dict[str, str]:
        """Get all resource attributes for OTel."""
        attrs = {
            "service.name": self.service_name,
            "deployment.environment": self.deployment_environment,
        }
        if self.service_namespace:
            attrs["service.namespace"] = self.service_namespace
        attrs.update(self.resource_attributes)
        return attrs


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
    otel_config: OtelConfig = field(default_factory=OtelConfig)
    failure_mode: FailureMode = FailureMode.FAIL_OPEN
    policy_refresh_interval_seconds: int = 300
    enable_budget_enforcement: bool = True
    enable_routing: bool = True
    enable_metrics: bool = True
    currency: str = "USD"
    default_max_iterations_per_run: int = 50
    default_max_tool_calls_per_run: int = 100
    default_max_tokens_per_run: int = 100000
    log_level: str = "INFO"
