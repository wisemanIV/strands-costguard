"""Budget specification and policy definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BudgetScope(str, Enum):
    """Scope at which a budget applies."""

    GLOBAL = "global"
    TENANT = "tenant"
    STRAND = "strand"
    WORKFLOW = "workflow"


class BudgetPeriod(str, Enum):
    """Time period for budget limits."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ThresholdAction(str, Enum):
    """Action to take when a soft threshold is exceeded."""

    LOG_ONLY = "LOG_ONLY"
    DOWNGRADE_MODEL = "DOWNGRADE_MODEL"
    LIMIT_CAPABILITIES = "LIMIT_CAPABILITIES"
    HALT_NEW_RUNS = "HALT_NEW_RUNS"


class HardLimitAction(str, Enum):
    """Action to take when a hard limit is exceeded."""

    HALT_RUN = "HALT_RUN"
    REJECT_NEW_RUNS = "REJECT_NEW_RUNS"


@dataclass
class BudgetMatch:
    """Criteria for matching a budget to a context."""

    tenant_id: str = "*"
    strand_id: str = "*"
    workflow_id: str = "*"

    def matches(self, tenant_id: str, strand_id: str, workflow_id: str) -> bool:
        """Check if this match criteria applies to the given context."""
        if self.tenant_id != "*" and self.tenant_id != tenant_id:
            return False
        if self.strand_id != "*" and self.strand_id != strand_id:
            return False
        if self.workflow_id != "*" and self.workflow_id != workflow_id:
            return False
        return True

    def specificity_score(self) -> int:
        """Calculate specificity score for priority ordering (higher = more specific)."""
        score = 0
        if self.tenant_id != "*":
            score += 1
        if self.strand_id != "*":
            score += 2
        if self.workflow_id != "*":
            score += 4
        return score


@dataclass
class BudgetConstraints:
    """Per-run constraints within a budget."""

    max_iterations_per_run: Optional[int] = None
    max_tool_calls_per_run: Optional[int] = None
    max_model_tokens_per_run: Optional[int] = None
    max_cost_per_run: Optional[float] = None


@dataclass
class BudgetSpec:
    """Complete budget specification."""

    id: str
    scope: BudgetScope
    match: BudgetMatch
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    max_cost: Optional[float] = None
    soft_thresholds: list[float] = field(default_factory=lambda: [0.7, 0.9, 1.0])
    hard_limit: bool = True
    on_soft_threshold_exceeded: ThresholdAction = ThresholdAction.LOG_ONLY
    on_hard_limit_exceeded: HardLimitAction = HardLimitAction.REJECT_NEW_RUNS
    max_runs_per_period: Optional[int] = None
    max_concurrent_runs: Optional[int] = None
    constraints: BudgetConstraints = field(default_factory=BudgetConstraints)
    enabled: bool = True

    def get_priority(self) -> int:
        """Get priority for policy merging (higher = higher priority)."""
        scope_priority = {
            BudgetScope.GLOBAL: 0,
            BudgetScope.TENANT: 10,
            BudgetScope.STRAND: 20,
            BudgetScope.WORKFLOW: 30,
        }
        return scope_priority.get(self.scope, 0) + self.match.specificity_score()

    def matches_context(self, tenant_id: str, strand_id: str, workflow_id: str) -> bool:
        """Check if this budget applies to the given context."""
        return self.enabled and self.match.matches(tenant_id, strand_id, workflow_id)

    def get_current_threshold_action(self, utilization: float) -> Optional[ThresholdAction]:
        """Get the action for the current budget utilization level."""
        exceeded_thresholds = [t for t in self.soft_thresholds if utilization >= t]
        if exceeded_thresholds:
            return self.on_soft_threshold_exceeded
        return None

    def is_hard_limit_exceeded(self, utilization: float) -> bool:
        """Check if the hard limit has been exceeded."""
        return self.hard_limit and utilization >= 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetSpec":
        """Create a BudgetSpec from a dictionary (e.g., parsed from YAML)."""
        match_data = data.get("match", {})
        match = BudgetMatch(
            tenant_id=match_data.get("tenant_id", "*"),
            strand_id=match_data.get("strand_id", "*"),
            workflow_id=match_data.get("workflow_id", "*"),
        )

        constraints_data = data.get("constraints", {})
        constraints = BudgetConstraints(
            max_iterations_per_run=constraints_data.get("max_iterations_per_run"),
            max_tool_calls_per_run=constraints_data.get("max_tool_calls_per_run"),
            max_model_tokens_per_run=constraints_data.get("max_model_tokens_per_run"),
            max_cost_per_run=constraints_data.get("max_cost_per_run"),
        )

        return cls(
            id=data["id"],
            scope=BudgetScope(data.get("scope", "tenant")),
            match=match,
            period=BudgetPeriod(data.get("period", "monthly")),
            max_cost=data.get("max_cost"),
            soft_thresholds=data.get("soft_thresholds", [0.7, 0.9, 1.0]),
            hard_limit=data.get("hard_limit", True),
            on_soft_threshold_exceeded=ThresholdAction(
                data.get("on_soft_threshold_exceeded", "LOG_ONLY")
            ),
            on_hard_limit_exceeded=HardLimitAction(
                data.get("on_hard_limit_exceeded", "REJECT_NEW_RUNS")
            ),
            max_runs_per_period=data.get("max_runs_per_period"),
            max_concurrent_runs=data.get("max_concurrent_runs"),
            constraints=constraints,
            enabled=data.get("enabled", True),
        )
