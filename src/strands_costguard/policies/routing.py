"""Routing policy definitions for adaptive model selection."""

from dataclasses import dataclass, field


class ModelStage:
    """Semantic stages for model calls within an agent."""

    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    SYNTHESIS = "synthesis"
    OTHER = "other"


@dataclass
class DowngradeTrigger:
    """Conditions that trigger model downgrade."""

    soft_threshold_exceeded: bool = False
    remaining_budget_below: float | None = None
    iteration_count_above: int | None = None
    latency_above_ms: float | None = None

    def should_downgrade(
        self,
        soft_threshold_exceeded: bool = False,
        remaining_budget: float | None = None,
        iteration_count: int = 0,
        avg_latency_ms: float | None = None,
    ) -> tuple[bool, str]:
        """Check if downgrade should be triggered, returning (should_downgrade, reason)."""
        if self.soft_threshold_exceeded and soft_threshold_exceeded:
            return True, "soft budget threshold exceeded"

        if self.remaining_budget_below is not None and remaining_budget is not None:
            if remaining_budget < self.remaining_budget_below:
                return (
                    True,
                    f"remaining budget ({remaining_budget:.2f}) below threshold ({self.remaining_budget_below:.2f})",
                )

        if self.iteration_count_above is not None:
            if iteration_count > self.iteration_count_above:
                return (
                    True,
                    f"iteration count ({iteration_count}) above threshold ({self.iteration_count_above})",
                )

        if self.latency_above_ms is not None and avg_latency_ms is not None:
            if avg_latency_ms > self.latency_above_ms:
                return (
                    True,
                    f"average latency ({avg_latency_ms:.0f}ms) above threshold ({self.latency_above_ms:.0f}ms)",
                )

        return False, ""

    @classmethod
    def from_dict(cls, data: dict) -> "DowngradeTrigger":
        """Create from dictionary."""
        return cls(
            soft_threshold_exceeded=data.get("soft_threshold_exceeded", False),
            remaining_budget_below=data.get("remaining_budget_below"),
            iteration_count_above=data.get("iteration_count_above"),
            latency_above_ms=data.get("latency_above_ms"),
        )


@dataclass
class StageConfig:
    """Configuration for a specific model call stage."""

    stage: str
    default_model: str
    fallback_model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    trigger_downgrade_on: DowngradeTrigger = field(default_factory=DowngradeTrigger)

    def get_effective_model(
        self,
        soft_threshold_exceeded: bool = False,
        remaining_budget: float | None = None,
        iteration_count: int = 0,
        avg_latency_ms: float | None = None,
    ) -> tuple[str, bool, str]:
        """
        Get the effective model to use given current conditions.

        Returns:
            Tuple of (model_name, was_downgraded, reason)
        """
        if self.fallback_model:
            should_downgrade, reason = self.trigger_downgrade_on.should_downgrade(
                soft_threshold_exceeded=soft_threshold_exceeded,
                remaining_budget=remaining_budget,
                iteration_count=iteration_count,
                avg_latency_ms=avg_latency_ms,
            )
            if should_downgrade:
                return self.fallback_model, True, reason

        return self.default_model, False, ""

    @classmethod
    def from_dict(cls, data: dict) -> "StageConfig":
        """Create from dictionary."""
        trigger_data = data.get("trigger_downgrade_on", {})
        return cls(
            stage=data["stage"],
            default_model=data["default_model"],
            fallback_model=data.get("fallback_model"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            trigger_downgrade_on=DowngradeTrigger.from_dict(trigger_data),
        )


@dataclass
class RoutingPolicy:
    """Complete routing policy for model selection."""

    id: str
    match: dict[str, str] = field(default_factory=dict)
    stages: list[StageConfig] = field(default_factory=list)
    default_model: str = "gpt-4o-mini"
    default_fallback_model: str | None = None
    enabled: bool = True

    def matches_context(self, tenant_id: str, strand_id: str, workflow_id: str) -> bool:
        """Check if this routing policy applies to the given context."""
        if not self.enabled:
            return False

        tenant_match = self.match.get("tenant_id", "*")
        strand_match = self.match.get("strand_id", "*")
        workflow_match = self.match.get("workflow_id", "*")

        if tenant_match != "*" and tenant_match != tenant_id:
            return False
        if strand_match != "*" and strand_match != strand_id:
            return False
        if workflow_match != "*" and workflow_match != workflow_id:
            return False

        return True

    def get_stage_config(self, stage: str) -> StageConfig | None:
        """Get configuration for a specific stage."""
        for stage_config in self.stages:
            if stage_config.stage == stage:
                return stage_config
        return None

    def get_model_for_stage(
        self,
        stage: str,
        soft_threshold_exceeded: bool = False,
        remaining_budget: float | None = None,
        iteration_count: int = 0,
        avg_latency_ms: float | None = None,
    ) -> tuple[str, int | None, bool, str]:
        """
        Get the effective model for a stage given current conditions.

        Returns:
            Tuple of (model_name, max_tokens, was_downgraded, reason)
        """
        stage_config = self.get_stage_config(stage)

        if stage_config:
            model, downgraded, reason = stage_config.get_effective_model(
                soft_threshold_exceeded=soft_threshold_exceeded,
                remaining_budget=remaining_budget,
                iteration_count=iteration_count,
                avg_latency_ms=avg_latency_ms,
            )
            return model, stage_config.max_tokens, downgraded, reason

        # Fall back to policy defaults
        return self.default_model, None, False, ""

    def specificity_score(self) -> int:
        """Calculate specificity score for priority ordering."""
        score = 0
        if self.match.get("tenant_id", "*") != "*":
            score += 1
        if self.match.get("strand_id", "*") != "*":
            score += 2
        if self.match.get("workflow_id", "*") != "*":
            score += 4
        return score

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingPolicy":
        """Create from dictionary."""
        stages = [StageConfig.from_dict(s) for s in data.get("stages", [])]
        return cls(
            id=data["id"],
            match=data.get("match", {}),
            stages=stages,
            default_model=data.get("default_model", "gpt-4o-mini"),
            default_fallback_model=data.get("default_fallback_model"),
            enabled=data.get("enabled", True),
        )
