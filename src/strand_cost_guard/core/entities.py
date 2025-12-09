"""Core entity identifiers for cost tracking and attribution."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass(frozen=True)
class RunContext:
    """Context for a single execution run, containing all identifiers for cost attribution."""

    tenant_id: str
    strand_id: str
    workflow_id: str
    run_id: str
    metadata: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> "RunContext":
        """Create a new run context, generating a run_id if not provided."""
        return cls(
            tenant_id=tenant_id,
            strand_id=strand_id,
            workflow_id=workflow_id,
            run_id=run_id or str(uuid.uuid4()),
            metadata=metadata or {},
        )

    def to_attributes(self) -> dict[str, str]:
        """Convert to OpenTelemetry attributes."""
        attrs = {
            "strands.tenant_id": self.tenant_id,
            "strands.strand_id": self.strand_id,
            "strands.workflow_id": self.workflow_id,
            "strands.run_id": self.run_id,
        }
        for key, value in self.metadata.items():
            attrs[f"strands.metadata.{key}"] = value
        return attrs


@dataclass
class RunState:
    """Mutable state for a single run, tracking costs and usage."""

    context: RunContext
    current_iteration: int = 0
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    model_costs: dict[str, float] = field(default_factory=dict)
    tool_costs: dict[str, float] = field(default_factory=dict)
    status: str = "running"
    ended_at: Optional[datetime] = None

    def add_model_cost(self, model_name: str, cost: float, input_tokens: int, output_tokens: int) -> None:
        """Record cost from a model call."""
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.model_costs[model_name] = self.model_costs.get(model_name, 0.0) + cost

    def add_tool_cost(self, tool_name: str, cost: float) -> None:
        """Record cost from a tool call."""
        self.total_cost += cost
        self.total_tool_calls += 1
        self.tool_costs[tool_name] = self.tool_costs.get(tool_name, 0.0) + cost

    def increment_iteration(self) -> int:
        """Increment and return the new iteration index."""
        self.current_iteration += 1
        return self.current_iteration

    def end(self, status: str) -> None:
        """Mark the run as ended with the given status."""
        self.status = status
        self.ended_at = datetime.utcnow()


@dataclass
class PeriodUsage:
    """Aggregated usage for a budget period (tenant/strand/workflow level)."""

    scope_type: str  # "tenant", "strand", "workflow"
    scope_id: str
    period_start: datetime
    period_end: datetime
    total_cost: float = 0.0
    total_runs: int = 0
    concurrent_runs: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0
    model_costs: dict[str, float] = field(default_factory=dict)
    tool_costs: dict[str, float] = field(default_factory=dict)

    def add_run_cost(self, run_state: RunState) -> None:
        """Add costs from a completed run to period totals."""
        self.total_cost += run_state.total_cost
        self.total_runs += 1
        self.total_input_tokens += run_state.total_input_tokens
        self.total_output_tokens += run_state.total_output_tokens
        self.total_iterations += run_state.current_iteration
        self.total_tool_calls += run_state.total_tool_calls

        for model, cost in run_state.model_costs.items():
            self.model_costs[model] = self.model_costs.get(model, 0.0) + cost
        for tool, cost in run_state.tool_costs.items():
            self.tool_costs[tool] = self.tool_costs.get(tool, 0.0) + cost

    def get_budget_utilization(self, max_cost: float) -> float:
        """Get budget utilization as a fraction (0.0 to 1.0+)."""
        if max_cost <= 0:
            return 0.0
        return self.total_cost / max_cost
