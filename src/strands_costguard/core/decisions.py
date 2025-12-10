"""Decision objects returned by Cost Guard lifecycle hooks."""

from dataclasses import dataclass, field
from enum import Enum


class DecisionAction(str, Enum):
    """Actions that can be taken based on budget/policy decisions."""

    ALLOW = "allow"
    REJECT = "reject"
    DOWNGRADE = "downgrade"
    LIMIT = "limit"
    HALT = "halt"
    LOG_ONLY = "log_only"


@dataclass
class ActionOverrides:
    """Overrides that can be applied to modify runtime behavior."""

    model_name: str | None = None
    max_tokens_remaining: int | None = None
    force_terminate_run: bool = False
    skip_tool_call: bool = False
    fallback_response: str | None = None


@dataclass
class AdmissionDecision:
    """Decision for whether to admit a new run."""

    allowed: bool
    reason: str | None = None
    action: DecisionAction = DecisionAction.ALLOW
    remaining_budget: float | None = None
    budget_utilization: float | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def admit(
        cls,
        remaining_budget: float | None = None,
        budget_utilization: float | None = None,
        warnings: list[str] | None = None,
    ) -> "AdmissionDecision":
        """Create an admission decision that allows the run."""
        return cls(
            allowed=True,
            action=DecisionAction.ALLOW,
            remaining_budget=remaining_budget,
            budget_utilization=budget_utilization,
            warnings=warnings or [],
        )

    @classmethod
    def reject(cls, reason: str) -> "AdmissionDecision":
        """Create an admission decision that rejects the run."""
        return cls(
            allowed=False,
            reason=reason,
            action=DecisionAction.REJECT,
        )


@dataclass
class IterationDecision:
    """Decision for whether to proceed with an iteration."""

    allowed: bool
    reason: str | None = None
    action: DecisionAction = DecisionAction.ALLOW
    action_overrides: ActionOverrides = field(default_factory=ActionOverrides)
    remaining_iterations: int | None = None
    remaining_budget: float | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def proceed(
        cls,
        remaining_iterations: int | None = None,
        remaining_budget: float | None = None,
        warnings: list[str] | None = None,
    ) -> "IterationDecision":
        """Create a decision to proceed with the iteration."""
        return cls(
            allowed=True,
            action=DecisionAction.ALLOW,
            remaining_iterations=remaining_iterations,
            remaining_budget=remaining_budget,
            warnings=warnings or [],
        )

    @classmethod
    def halt(cls, reason: str) -> "IterationDecision":
        """Create a decision to halt the run."""
        return cls(
            allowed=False,
            reason=reason,
            action=DecisionAction.HALT,
            action_overrides=ActionOverrides(force_terminate_run=True),
        )


@dataclass
class ModelDecision:
    """Decision for a model call, including potential model routing."""

    allowed: bool
    reason: str | None = None
    action: DecisionAction = DecisionAction.ALLOW
    action_overrides: ActionOverrides = field(default_factory=ActionOverrides)
    effective_model: str | None = None
    max_tokens: int | None = None
    remaining_tokens: int | None = None
    remaining_budget: float | None = None
    was_downgraded: bool = False
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def allow(
        cls,
        effective_model: str,
        max_tokens: int | None = None,
        remaining_budget: float | None = None,
        warnings: list[str] | None = None,
    ) -> "ModelDecision":
        """Create a decision to allow the model call."""
        return cls(
            allowed=True,
            action=DecisionAction.ALLOW,
            effective_model=effective_model,
            max_tokens=max_tokens,
            remaining_budget=remaining_budget,
            warnings=warnings or [],
        )

    @classmethod
    def downgrade(
        cls,
        original_model: str,
        fallback_model: str,
        reason: str,
        max_tokens: int | None = None,
    ) -> "ModelDecision":
        """Create a decision to downgrade to a fallback model."""
        return cls(
            allowed=True,
            reason=reason,
            action=DecisionAction.DOWNGRADE,
            action_overrides=ActionOverrides(model_name=fallback_model),
            effective_model=fallback_model,
            max_tokens=max_tokens,
            was_downgraded=True,
            warnings=[f"Downgraded from {original_model} to {fallback_model}: {reason}"],
        )

    @classmethod
    def reject(cls, reason: str) -> "ModelDecision":
        """Create a decision to reject the model call."""
        return cls(
            allowed=False,
            reason=reason,
            action=DecisionAction.REJECT,
        )


@dataclass
class ToolDecision:
    """Decision for a tool call."""

    allowed: bool
    reason: str | None = None
    action: DecisionAction = DecisionAction.ALLOW
    action_overrides: ActionOverrides = field(default_factory=ActionOverrides)
    remaining_tool_calls: int | None = None
    remaining_budget: float | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def allow(
        cls,
        remaining_tool_calls: int | None = None,
        remaining_budget: float | None = None,
        warnings: list[str] | None = None,
    ) -> "ToolDecision":
        """Create a decision to allow the tool call."""
        return cls(
            allowed=True,
            action=DecisionAction.ALLOW,
            remaining_tool_calls=remaining_tool_calls,
            remaining_budget=remaining_budget,
            warnings=warnings or [],
        )

    @classmethod
    def reject(cls, reason: str) -> "ToolDecision":
        """Create a decision to reject the tool call."""
        return cls(
            allowed=False,
            reason=reason,
            action=DecisionAction.REJECT,
            action_overrides=ActionOverrides(skip_tool_call=True),
        )
