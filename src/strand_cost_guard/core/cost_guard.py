"""Main CostGuard runtime with lifecycle hooks."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from strand_cost_guard.core.budget_tracker import BudgetTracker
from strand_cost_guard.core.config import CostGuardConfig
from strand_cost_guard.core.decisions import (
    AdmissionDecision,
    DecisionAction,
    IterationDecision,
    ModelDecision,
    ToolDecision,
)
from strand_cost_guard.core.entities import RunContext, RunState
from strand_cost_guard.core.usage import IterationUsage, ModelUsage, ToolUsage
from strand_cost_guard.metrics.otel import MetricsEmitter
from strand_cost_guard.policies.budget import BudgetSpec, HardLimitAction, ThresholdAction
from strand_cost_guard.policies.store import PolicyStore
from strand_cost_guard.pricing.table import PricingTable

logger = logging.getLogger(__name__)


@dataclass
class CostGuard:
    """
    Main Cost Guard component for budget enforcement and cost tracking.

    Integrates with Strands runtime via lifecycle hooks to:
    - Enforce budget limits at run, iteration, and call levels
    - Track and attribute costs by tenant, strand, workflow, run, model, and tool
    - Emit OpenTelemetry-compatible metrics
    - Support adaptive model routing based on budget state
    """

    config: CostGuardConfig
    _policy_store: PolicyStore = field(init=False)
    _pricing_table: PricingTable = field(init=False)
    _budget_tracker: BudgetTracker = field(init=False)
    _metrics_emitter: Optional[MetricsEmitter] = field(init=False, default=None)
    _run_budgets: dict[str, list[BudgetSpec]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize internal components."""
        self._policy_store = PolicyStore(
            source=self.config.policy_source,
            refresh_interval_seconds=self.config.policy_refresh_interval_seconds,
        )

        pricing_data = self._policy_store.get_pricing()
        if pricing_data:
            self._pricing_table = PricingTable.from_dict(pricing_data)
        else:
            self._pricing_table = PricingTable(currency=self.config.currency)

        self._budget_tracker = BudgetTracker()

        if self.config.enable_metrics:
            self._metrics_emitter = MetricsEmitter(
                include_run_id=self.config.include_run_id_in_metrics
            )

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    def on_run_start(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
        run_id: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> AdmissionDecision:
        """
        Called when a new run starts.

        Returns:
            AdmissionDecision indicating whether the run should proceed.
        """
        context = RunContext.create(
            tenant_id=tenant_id,
            strand_id=strand_id,
            workflow_id=workflow_id,
            run_id=run_id,
            metadata=metadata,
        )

        # Get applicable budgets
        budgets = self._policy_store.get_budgets_for_context(
            tenant_id, strand_id, workflow_id
        )
        self._run_budgets[run_id] = budgets

        if not self.config.enable_budget_enforcement:
            run_state = RunState(context=context)
            self._budget_tracker.register_run(run_state, budgets)
            return AdmissionDecision.admit()

        # Check for exceeded limits
        exceeded = self._budget_tracker.check_budget_limits(
            tenant_id, strand_id, workflow_id, budgets
        )

        if exceeded:
            for budget, state, reason in exceeded:
                if budget.on_hard_limit_exceeded == HardLimitAction.REJECT_NEW_RUNS:
                    self._emit_rejection_event(context, reason)
                    return AdmissionDecision.reject(reason)

        # Register the run
        run_state = RunState(context=context)
        self._budget_tracker.register_run(run_state, budgets)

        # Calculate remaining budget and utilization
        remaining = None
        utilization = None
        warnings = []

        for budget in budgets:
            state = self._budget_tracker.get_or_create_budget_state(
                budget, tenant_id, strand_id, workflow_id
            )
            if budget.max_cost:
                if remaining is None or (state.remaining_budget and state.remaining_budget < remaining):
                    remaining = state.remaining_budget
                    utilization = state.utilization

            # Check soft thresholds for warnings
            threshold_action = budget.get_current_threshold_action(state.utilization)
            if threshold_action and threshold_action != ThresholdAction.LOG_ONLY:
                warnings.append(
                    f"Budget {budget.id} at {state.utilization:.1%} utilization"
                )

        self._emit_run_start(context)

        return AdmissionDecision.admit(
            remaining_budget=remaining,
            budget_utilization=utilization,
            warnings=warnings,
        )

    def on_run_end(self, run_id: str, status: str) -> None:
        """
        Called when a run ends.

        Args:
            run_id: The run identifier
            status: Final status (e.g., "completed", "failed", "cancelled")
        """
        budgets = self._run_budgets.pop(run_id, [])
        run_state = self._budget_tracker.unregister_run(run_id, budgets)

        if run_state:
            run_state.end(status)
            self._emit_run_end(run_state)

    def before_iteration(
        self,
        run_id: str,
        iteration_idx: int,
        context: Optional[dict[str, Any]] = None,
    ) -> IterationDecision:
        """
        Called before each agent loop iteration.

        Args:
            run_id: The run identifier
            iteration_idx: Current iteration index (0-based)
            context: Optional context about the iteration

        Returns:
            IterationDecision indicating whether to proceed.
        """
        run_state = self._budget_tracker.get_run_state(run_id)
        if not run_state:
            logger.warning(f"before_iteration called for unknown run: {run_id}")
            return IterationDecision.proceed()

        budgets = self._run_budgets.get(run_id, [])
        warnings = []

        # Check iteration limits from constraints
        for budget in budgets:
            max_iterations = budget.constraints.max_iterations_per_run
            if max_iterations is None:
                max_iterations = self.config.default_max_iterations_per_run

            if iteration_idx >= max_iterations:
                reason = f"Max iterations ({max_iterations}) exceeded"
                self._emit_iteration_halt(run_state, reason)
                return IterationDecision.halt(reason)

        # Check budget utilization
        if self.config.enable_budget_enforcement:
            for budget in budgets:
                state = self._budget_tracker.get_or_create_budget_state(
                    budget,
                    run_state.context.tenant_id,
                    run_state.context.strand_id,
                    run_state.context.workflow_id,
                )

                if budget.is_hard_limit_exceeded(state.utilization):
                    if budget.on_hard_limit_exceeded == HardLimitAction.HALT_RUN:
                        reason = f"Budget {budget.id} hard limit exceeded during run"
                        self._emit_iteration_halt(run_state, reason)
                        return IterationDecision.halt(reason)

                threshold_action = budget.get_current_threshold_action(state.utilization)
                if threshold_action:
                    warnings.append(f"Budget {budget.id} at {state.utilization:.1%}")

        # Calculate remaining iterations
        remaining = None
        for budget in budgets:
            max_iter = budget.constraints.max_iterations_per_run or self.config.default_max_iterations_per_run
            budget_remaining = max_iter - iteration_idx
            if remaining is None or budget_remaining < remaining:
                remaining = budget_remaining

        return IterationDecision.proceed(
            remaining_iterations=remaining,
            warnings=warnings,
        )

    def after_iteration(
        self,
        run_id: str,
        iteration_idx: int,
        usage: IterationUsage,
    ) -> None:
        """
        Called after each agent loop iteration completes.

        Args:
            run_id: The run identifier
            iteration_idx: Completed iteration index
            usage: Usage metrics for the iteration
        """
        run_state = self._budget_tracker.get_run_state(run_id)
        if run_state:
            run_state.current_iteration = iteration_idx + 1
            self._emit_iteration_complete(run_state, usage)

    def before_model_call(
        self,
        run_id: str,
        model_name: str,
        stage: str = "other",
        prompt_tokens_estimate: int = 0,
    ) -> ModelDecision:
        """
        Called before each model call.

        Args:
            run_id: The run identifier
            model_name: Requested model name
            stage: Semantic stage ("planning", "tool_selection", "synthesis", "other")
            prompt_tokens_estimate: Estimated prompt tokens

        Returns:
            ModelDecision with effective model and any overrides.
        """
        run_state = self._budget_tracker.get_run_state(run_id)
        if not run_state:
            return ModelDecision.allow(effective_model=model_name)

        budgets = self._run_budgets.get(run_id, [])
        effective_model = model_name
        max_tokens = None
        was_downgraded = False
        downgrade_reason = ""
        warnings = []

        # Check routing policy for model selection
        if self.config.enable_routing:
            routing_policy = self._policy_store.get_routing_policy(
                run_state.context.tenant_id,
                run_state.context.strand_id,
                run_state.context.workflow_id,
            )

            if routing_policy:
                # Determine if we should downgrade based on budget state
                soft_threshold_exceeded = False
                remaining_budget = None

                for budget in budgets:
                    state = self._budget_tracker.get_or_create_budget_state(
                        budget,
                        run_state.context.tenant_id,
                        run_state.context.strand_id,
                        run_state.context.workflow_id,
                    )
                    threshold_action = budget.get_current_threshold_action(state.utilization)
                    if threshold_action == ThresholdAction.DOWNGRADE_MODEL:
                        soft_threshold_exceeded = True

                    if state.remaining_budget is not None:
                        if remaining_budget is None or state.remaining_budget < remaining_budget:
                            remaining_budget = state.remaining_budget

                effective_model, max_tokens, was_downgraded, downgrade_reason = (
                    routing_policy.get_model_for_stage(
                        stage=stage,
                        soft_threshold_exceeded=soft_threshold_exceeded,
                        remaining_budget=remaining_budget,
                        iteration_count=run_state.current_iteration,
                    )
                )

        # Check token limits
        for budget in budgets:
            max_run_tokens = budget.constraints.max_model_tokens_per_run
            if max_run_tokens:
                remaining_tokens = max_run_tokens - run_state.total_input_tokens - run_state.total_output_tokens
                if remaining_tokens <= 0:
                    return ModelDecision.reject(f"Token limit ({max_run_tokens}) exceeded for run")
                if max_tokens is None or remaining_tokens < max_tokens:
                    max_tokens = remaining_tokens

        # Estimate cost and check budget
        if prompt_tokens_estimate > 0 and self.config.enable_budget_enforcement:
            estimated_cost = self._pricing_table.estimate_model_cost(
                effective_model, prompt_tokens_estimate
            )
            for budget in budgets:
                state = self._budget_tracker.get_or_create_budget_state(
                    budget,
                    run_state.context.tenant_id,
                    run_state.context.strand_id,
                    run_state.context.workflow_id,
                )
                if state.remaining_budget is not None and estimated_cost > state.remaining_budget:
                    warnings.append(
                        f"Estimated cost (${estimated_cost:.4f}) exceeds remaining budget (${state.remaining_budget:.4f})"
                    )

        if was_downgraded:
            self._emit_downgrade_event(run_state, model_name, effective_model, downgrade_reason)
            return ModelDecision.downgrade(
                original_model=model_name,
                fallback_model=effective_model,
                reason=downgrade_reason,
                max_tokens=max_tokens,
            )

        return ModelDecision.allow(
            effective_model=effective_model,
            max_tokens=max_tokens,
            warnings=warnings,
        )

    def after_model_call(
        self,
        run_id: str,
        usage: ModelUsage,
    ) -> None:
        """
        Called after each model call completes.

        Args:
            run_id: The run identifier
            usage: Usage metrics from the model call
        """
        # Calculate cost if not already set
        if usage.cost == 0:
            usage.cost = self._pricing_table.calculate_model_cost(
                model_name=usage.model_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )

        self._budget_tracker.update_run_cost(
            run_id=run_id,
            model_name=usage.model_name,
            model_cost=usage.cost,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

        run_state = self._budget_tracker.get_run_state(run_id)
        if run_state:
            self._emit_model_cost(run_state, usage)

    def before_tool_call(
        self,
        run_id: str,
        tool_name: str,
    ) -> ToolDecision:
        """
        Called before each tool call.

        Args:
            run_id: The run identifier
            tool_name: Name of the tool being called

        Returns:
            ToolDecision indicating whether to proceed.
        """
        run_state = self._budget_tracker.get_run_state(run_id)
        if not run_state:
            return ToolDecision.allow()

        budgets = self._run_budgets.get(run_id, [])
        warnings = []

        # Check tool call limits
        for budget in budgets:
            max_calls = budget.constraints.max_tool_calls_per_run
            if max_calls is None:
                max_calls = self.config.default_max_tool_calls_per_run

            if run_state.total_tool_calls >= max_calls:
                reason = f"Max tool calls ({max_calls}) exceeded"
                return ToolDecision.reject(reason)

        # Calculate remaining tool calls
        remaining = None
        for budget in budgets:
            max_calls = budget.constraints.max_tool_calls_per_run or self.config.default_max_tool_calls_per_run
            budget_remaining = max_calls - run_state.total_tool_calls
            if remaining is None or budget_remaining < remaining:
                remaining = budget_remaining

        return ToolDecision.allow(
            remaining_tool_calls=remaining,
            warnings=warnings,
        )

    def after_tool_call(
        self,
        run_id: str,
        tool_name: str,
        cost_metadata: ToolUsage,
    ) -> None:
        """
        Called after each tool call completes.

        Args:
            run_id: The run identifier
            tool_name: Name of the tool called
            cost_metadata: Usage and cost metadata from the tool
        """
        # Calculate cost if not already set
        if cost_metadata.cost == 0:
            cost_metadata.cost = self._pricing_table.calculate_tool_cost(
                tool_name=tool_name,
                input_size_bytes=cost_metadata.input_size_bytes,
                output_size_bytes=cost_metadata.output_size_bytes,
            )

        self._budget_tracker.update_run_cost(
            run_id=run_id,
            tool_name=tool_name,
            tool_cost=cost_metadata.cost,
        )

        run_state = self._budget_tracker.get_run_state(run_id)
        if run_state:
            self._emit_tool_cost(run_state, cost_metadata)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_run_cost(self, run_id: str) -> Optional[float]:
        """Get the current total cost for a run."""
        run_state = self._budget_tracker.get_run_state(run_id)
        return run_state.total_cost if run_state else None

    def get_budget_summary(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
    ) -> dict[str, dict]:
        """Get a summary of budget usage for a context."""
        budgets = self._policy_store.get_budgets_for_context(
            tenant_id, strand_id, workflow_id
        )
        return self._budget_tracker.get_budget_summary(
            tenant_id, strand_id, workflow_id, budgets
        )

    # =========================================================================
    # Metrics Emission (Internal)
    # =========================================================================

    def _emit_run_start(self, context: RunContext) -> None:
        """Emit metrics for run start."""
        if self._metrics_emitter:
            self._metrics_emitter.record_run_start(context)

    def _emit_run_end(self, run_state: RunState) -> None:
        """Emit metrics for run end."""
        if self._metrics_emitter:
            self._metrics_emitter.record_run_end(run_state)

    def _emit_model_cost(self, run_state: RunState, usage: ModelUsage) -> None:
        """Emit metrics for model cost."""
        if self._metrics_emitter:
            self._metrics_emitter.record_model_cost(run_state.context, usage)

    def _emit_tool_cost(self, run_state: RunState, usage: ToolUsage) -> None:
        """Emit metrics for tool cost."""
        if self._metrics_emitter:
            self._metrics_emitter.record_tool_cost(run_state.context, usage)

    def _emit_downgrade_event(
        self,
        run_state: RunState,
        original_model: str,
        fallback_model: str,
        reason: str,
    ) -> None:
        """Emit metrics for model downgrade."""
        if self._metrics_emitter:
            self._metrics_emitter.record_downgrade(
                run_state.context, original_model, fallback_model, reason
            )

    def _emit_rejection_event(self, context: RunContext, reason: str) -> None:
        """Emit metrics for run rejection."""
        if self._metrics_emitter:
            self._metrics_emitter.record_rejection(context, reason)

    def _emit_iteration_halt(self, run_state: RunState, reason: str) -> None:
        """Emit metrics for iteration halt."""
        if self._metrics_emitter:
            self._metrics_emitter.record_iteration_halt(run_state.context, reason)

    def _emit_iteration_complete(self, run_state: RunState, usage: IterationUsage) -> None:
        """Emit metrics for iteration completion."""
        if self._metrics_emitter:
            self._metrics_emitter.record_iteration(run_state.context, usage)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def shutdown(self) -> None:
        """Shutdown the Cost Guard.

        Note: Metrics flushing is handled by StrandsTelemetry shutdown.
        Call this method for any Cost Guard-specific cleanup.
        """
        # No-op: MeterProvider lifecycle is managed by StrandsTelemetry
        pass
