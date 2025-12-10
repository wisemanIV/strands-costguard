"""Budget tracking and period management."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import TYPE_CHECKING, Optional

from strands_costguard.core.entities import PeriodUsage, RunState
from strands_costguard.policies.budget import BudgetPeriod, BudgetScope, BudgetSpec

if TYPE_CHECKING:
    from strands_costguard.persistence.valkey_store import ValkeyBudgetStore

logger = logging.getLogger(__name__)


def get_period_boundaries(
    period: BudgetPeriod, reference_time: datetime | None = None
) -> tuple[datetime, datetime]:
    """Calculate the start and end times for a budget period."""
    now = reference_time or datetime.utcnow()

    if period == BudgetPeriod.HOURLY:
        start = now.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
    elif period == BudgetPeriod.DAILY:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif period == BudgetPeriod.WEEKLY:
        # Week starts on Monday
        days_since_monday = now.weekday()
        start = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(weeks=1)
    elif period == BudgetPeriod.MONTHLY:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Calculate next month
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)
    else:
        raise ValueError(f"Unknown period: {period}")

    return start, end


@dataclass
class BudgetState:
    """Current state of a budget for a specific scope."""

    budget: BudgetSpec
    usage: PeriodUsage
    active_runs: set[str] = field(default_factory=set)

    @property
    def utilization(self) -> float:
        """Get current budget utilization (0.0 to 1.0+)."""
        if self.budget.max_cost is None or self.budget.max_cost <= 0:
            return 0.0
        return self.usage.total_cost / self.budget.max_cost

    @property
    def remaining_budget(self) -> float | None:
        """Get remaining budget, or None if unlimited."""
        if self.budget.max_cost is None:
            return None
        return max(0.0, self.budget.max_cost - self.usage.total_cost)

    @property
    def concurrent_runs(self) -> int:
        """Get number of currently active runs."""
        return len(self.active_runs)

    def is_period_expired(self, now: datetime | None = None) -> bool:
        """Check if the current period has expired."""
        now = now or datetime.utcnow()
        return now >= self.usage.period_end

    def reset_for_new_period(self, now: datetime | None = None) -> None:
        """Reset usage for a new period."""
        now = now or datetime.utcnow()
        start, end = get_period_boundaries(self.budget.period, now)
        self.usage = PeriodUsage(
            scope_type=self.budget.scope.value,
            scope_id=self.budget.id,
            period_start=start,
            period_end=end,
        )
        # Keep active runs - they may have started in previous period


@dataclass
class BudgetTracker:
    """
    Tracks budget usage across all scopes with thread-safe operations.

    Maintains in-memory counters for fast access. When a ValkeyBudgetStore is
    provided, state is persisted to Valkey/Redis and survives restarts.

    Usage with persistence:
        import valkey
        from strands_costguard.persistence import ValkeyBudgetStore

        client = valkey.Valkey(host="localhost", port=6379)
        store = ValkeyBudgetStore(client)
        tracker = BudgetTracker(store=store)
    """

    store: Optional["ValkeyBudgetStore"] = None
    _lock: Lock = field(default_factory=Lock)
    _budget_states: dict[str, BudgetState] = field(default_factory=dict)
    _run_states: dict[str, RunState] = field(default_factory=dict)

    def _get_scope_key(
        self, budget: BudgetSpec, tenant_id: str, strand_id: str, workflow_id: str
    ) -> str:
        """Generate a unique key for a budget scope."""
        if budget.scope == BudgetScope.GLOBAL:
            return f"global:{budget.id}"
        elif budget.scope == BudgetScope.TENANT:
            return f"tenant:{tenant_id}:{budget.id}"
        elif budget.scope == BudgetScope.STRAND:
            return f"strand:{tenant_id}:{strand_id}:{budget.id}"
        elif budget.scope == BudgetScope.WORKFLOW:
            return f"workflow:{tenant_id}:{strand_id}:{workflow_id}:{budget.id}"
        else:
            return f"unknown:{budget.id}"

    def get_or_create_budget_state(
        self,
        budget: BudgetSpec,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
    ) -> BudgetState:
        """Get or create budget state for a scope, resetting if period expired.

        If a store is configured, state is loaded from/persisted to Valkey.
        """
        key = self._get_scope_key(budget, tenant_id, strand_id, workflow_id)

        with self._lock:
            # Check in-memory cache first
            if key in self._budget_states:
                state = self._budget_states[key]
                if state.is_period_expired():
                    logger.info(f"Budget period expired for {key}, resetting")
                    state.reset_for_new_period()
                    self._persist_state(key, state)
                return state

            # Try to load from persistent store
            start, end = get_period_boundaries(budget.period)
            if self.store:
                persisted = self.store.get_or_create(
                    scope_key=key,
                    budget_id=budget.id,
                    period_start=start,
                    period_end=end,
                )
                # Convert persisted data to BudgetState
                usage = PeriodUsage(
                    scope_type=budget.scope.value,
                    scope_id=budget.id,
                    period_start=datetime.fromisoformat(persisted.period_start),
                    period_end=datetime.fromisoformat(persisted.period_end),
                    total_cost=persisted.total_cost,
                    total_runs=persisted.total_runs,
                    total_input_tokens=persisted.total_input_tokens,
                    total_output_tokens=persisted.total_output_tokens,
                    total_iterations=persisted.total_iterations,
                    total_tool_calls=persisted.total_tool_calls,
                    model_costs=persisted.model_costs,
                    tool_costs=persisted.tool_costs,
                    concurrent_runs=len(persisted.concurrent_run_ids),
                )
                state = BudgetState(
                    budget=budget,
                    usage=usage,
                    active_runs=set(persisted.concurrent_run_ids),
                )
                self._budget_states[key] = state
                return state

            # Create new in-memory state
            usage = PeriodUsage(
                scope_type=budget.scope.value,
                scope_id=budget.id,
                period_start=start,
                period_end=end,
            )
            state = BudgetState(budget=budget, usage=usage)
            self._budget_states[key] = state
            return state

    def _persist_state(self, key: str, state: BudgetState) -> None:
        """Persist budget state to store if configured."""
        if not self.store:
            return

        from strands_costguard.persistence.valkey_store import BudgetStateData

        data = BudgetStateData(
            budget_id=state.budget.id,
            scope_key=key,
            period_start=state.usage.period_start.isoformat(),
            period_end=state.usage.period_end.isoformat(),
            total_cost=state.usage.total_cost,
            total_runs=state.usage.total_runs,
            total_input_tokens=state.usage.total_input_tokens,
            total_output_tokens=state.usage.total_output_tokens,
            total_iterations=state.usage.total_iterations,
            total_tool_calls=state.usage.total_tool_calls,
            model_costs=state.usage.model_costs,
            tool_costs=state.usage.tool_costs,
            concurrent_run_ids=list(state.active_runs),
        )
        self.store.set(key, data, expire_at=state.usage.period_end)

    def register_run(
        self,
        run_state: RunState,
        budgets: list[BudgetSpec],
    ) -> None:
        """Register a new run and update concurrent run counts."""
        with self._lock:
            self._run_states[run_state.context.run_id] = run_state

            for budget in budgets:
                key = self._get_scope_key(
                    budget,
                    run_state.context.tenant_id,
                    run_state.context.strand_id,
                    run_state.context.workflow_id,
                )
                if key in self._budget_states:
                    self._budget_states[key].active_runs.add(run_state.context.run_id)
                    self._budget_states[key].usage.concurrent_runs = len(
                        self._budget_states[key].active_runs
                    )

                # Persist run registration
                if self.store:
                    self.store.increment_run_count(key, run_state.context.run_id)

    def unregister_run(self, run_id: str, budgets: list[BudgetSpec]) -> RunState | None:
        """Unregister a completed run and update totals."""
        with self._lock:
            run_state = self._run_states.pop(run_id, None)
            if not run_state:
                return None

            for budget in budgets:
                key = self._get_scope_key(
                    budget,
                    run_state.context.tenant_id,
                    run_state.context.strand_id,
                    run_state.context.workflow_id,
                )
                if key in self._budget_states:
                    state = self._budget_states[key]
                    state.active_runs.discard(run_id)
                    state.usage.concurrent_runs = len(state.active_runs)
                    state.usage.add_run_cost(run_state)
                    # Persist updated state
                    self._persist_state(key, state)

                # Also update persistent store directly
                if self.store:
                    self.store.remove_concurrent_run(key, run_id)

            return run_state

    def get_run_state(self, run_id: str) -> RunState | None:
        """Get the current state of a run."""
        with self._lock:
            return self._run_states.get(run_id)

    def update_run_cost(
        self,
        run_id: str,
        model_name: str | None = None,
        model_cost: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_name: str | None = None,
        tool_cost: float = 0.0,
    ) -> None:
        """Update costs for a run (does not update period totals until run ends)."""
        with self._lock:
            run_state = self._run_states.get(run_id)
            if not run_state:
                logger.warning(f"Attempted to update cost for unknown run: {run_id}")
                return

            if model_name and model_cost > 0:
                run_state.add_model_cost(model_name, model_cost, input_tokens, output_tokens)

            if tool_name:
                run_state.add_tool_cost(tool_name, tool_cost)

    def check_budget_limits(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
        budgets: list[BudgetSpec],
    ) -> list[tuple[BudgetSpec, BudgetState, str]]:
        """
        Check all applicable budgets and return any that are exceeded.

        Returns:
            List of (budget, state, reason) tuples for exceeded limits.
        """
        exceeded = []

        for budget in budgets:
            state = self.get_or_create_budget_state(budget, tenant_id, strand_id, workflow_id)

            # Check hard cost limit
            if budget.max_cost and state.utilization >= 1.0 and budget.hard_limit:
                exceeded.append(
                    (
                        budget,
                        state,
                        f"Budget {budget.id} hard limit exceeded: {state.utilization:.1%}",
                    )
                )
                continue

            # Check max runs per period
            if budget.max_runs_per_period and state.usage.total_runs >= budget.max_runs_per_period:
                exceeded.append(
                    (
                        budget,
                        state,
                        f"Budget {budget.id} max runs exceeded: {state.usage.total_runs}/{budget.max_runs_per_period}",
                    )
                )
                continue

            # Check concurrent runs
            if budget.max_concurrent_runs and state.concurrent_runs >= budget.max_concurrent_runs:
                exceeded.append(
                    (
                        budget,
                        state,
                        f"Budget {budget.id} max concurrent runs exceeded: {state.concurrent_runs}/{budget.max_concurrent_runs}",
                    )
                )

        return exceeded

    def get_budget_summary(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
        budgets: list[BudgetSpec],
    ) -> dict[str, dict]:
        """Get a summary of budget usage for all applicable budgets."""
        summary = {}

        for budget in budgets:
            state = self.get_or_create_budget_state(budget, tenant_id, strand_id, workflow_id)
            summary[budget.id] = {
                "scope": budget.scope.value,
                "period": budget.period.value,
                "max_cost": budget.max_cost,
                "current_cost": state.usage.total_cost,
                "utilization": state.utilization,
                "remaining": state.remaining_budget,
                "total_runs": state.usage.total_runs,
                "concurrent_runs": state.concurrent_runs,
                "period_start": state.usage.period_start.isoformat(),
                "period_end": state.usage.period_end.isoformat(),
            }

        return summary
