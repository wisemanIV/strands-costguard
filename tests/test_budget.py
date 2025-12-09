"""Tests for budget policies and tracking."""

import pytest
from datetime import datetime, timedelta

from strand_cost_guard.policies.budget import (
    BudgetSpec,
    BudgetScope,
    BudgetPeriod,
    BudgetMatch,
    BudgetConstraints,
    ThresholdAction,
    HardLimitAction,
)
from strand_cost_guard.core.budget_tracker import (
    BudgetTracker,
    get_period_boundaries,
)
from strand_cost_guard.core.entities import RunContext, RunState


class TestBudgetMatch:
    """Tests for BudgetMatch."""

    def test_wildcard_matches_all(self):
        match = BudgetMatch(tenant_id="*", strand_id="*", workflow_id="*")
        assert match.matches("tenant-1", "strand-1", "workflow-1")
        assert match.matches("any", "thing", "works")

    def test_specific_tenant_match(self):
        match = BudgetMatch(tenant_id="tenant-1", strand_id="*", workflow_id="*")
        assert match.matches("tenant-1", "strand-1", "workflow-1")
        assert not match.matches("tenant-2", "strand-1", "workflow-1")

    def test_specific_strand_match(self):
        match = BudgetMatch(tenant_id="*", strand_id="analytics", workflow_id="*")
        assert match.matches("tenant-1", "analytics", "workflow-1")
        assert not match.matches("tenant-1", "codegen", "workflow-1")

    def test_full_specific_match(self):
        match = BudgetMatch(
            tenant_id="tenant-1",
            strand_id="analytics",
            workflow_id="report",
        )
        assert match.matches("tenant-1", "analytics", "report")
        assert not match.matches("tenant-1", "analytics", "other")

    def test_specificity_score(self):
        # Wildcard has lowest score
        assert BudgetMatch().specificity_score() == 0

        # Tenant only
        assert BudgetMatch(tenant_id="t1").specificity_score() == 1

        # Strand only (higher priority)
        assert BudgetMatch(strand_id="s1").specificity_score() == 2

        # Workflow only (highest single)
        assert BudgetMatch(workflow_id="w1").specificity_score() == 4

        # Combined
        assert BudgetMatch(tenant_id="t1", strand_id="s1").specificity_score() == 3
        assert BudgetMatch(tenant_id="t1", strand_id="s1", workflow_id="w1").specificity_score() == 7


class TestBudgetSpec:
    """Tests for BudgetSpec."""

    def test_from_dict_minimal(self):
        data = {
            "id": "test-budget",
            "scope": "tenant",
            "match": {"tenant_id": "*"},
        }
        budget = BudgetSpec.from_dict(data)

        assert budget.id == "test-budget"
        assert budget.scope == BudgetScope.TENANT
        assert budget.period == BudgetPeriod.MONTHLY
        assert budget.max_cost is None
        assert budget.hard_limit is True

    def test_from_dict_full(self):
        data = {
            "id": "full-budget",
            "scope": "strand",
            "match": {"tenant_id": "t1", "strand_id": "s1"},
            "period": "daily",
            "max_cost": 100.0,
            "soft_thresholds": [0.5, 0.8],
            "hard_limit": True,
            "on_soft_threshold_exceeded": "DOWNGRADE_MODEL",
            "on_hard_limit_exceeded": "HALT_RUN",
            "max_runs_per_period": 500,
            "constraints": {
                "max_iterations_per_run": 10,
                "max_tool_calls_per_run": 20,
            },
        }
        budget = BudgetSpec.from_dict(data)

        assert budget.id == "full-budget"
        assert budget.scope == BudgetScope.STRAND
        assert budget.period == BudgetPeriod.DAILY
        assert budget.max_cost == 100.0
        assert budget.soft_thresholds == [0.5, 0.8]
        assert budget.on_soft_threshold_exceeded == ThresholdAction.DOWNGRADE_MODEL
        assert budget.on_hard_limit_exceeded == HardLimitAction.HALT_RUN
        assert budget.constraints.max_iterations_per_run == 10

    def test_priority_ordering(self):
        global_budget = BudgetSpec(
            id="global",
            scope=BudgetScope.GLOBAL,
            match=BudgetMatch(),
        )
        tenant_budget = BudgetSpec(
            id="tenant",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(tenant_id="t1"),
        )
        strand_budget = BudgetSpec(
            id="strand",
            scope=BudgetScope.STRAND,
            match=BudgetMatch(strand_id="s1"),
        )
        workflow_budget = BudgetSpec(
            id="workflow",
            scope=BudgetScope.WORKFLOW,
            match=BudgetMatch(workflow_id="w1"),
        )

        assert global_budget.get_priority() < tenant_budget.get_priority()
        assert tenant_budget.get_priority() < strand_budget.get_priority()
        assert strand_budget.get_priority() < workflow_budget.get_priority()

    def test_threshold_action(self):
        budget = BudgetSpec(
            id="test",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(),
            soft_thresholds=[0.7, 0.9],
            on_soft_threshold_exceeded=ThresholdAction.DOWNGRADE_MODEL,
        )

        # Below all thresholds
        assert budget.get_current_threshold_action(0.5) is None

        # At first threshold
        assert budget.get_current_threshold_action(0.7) == ThresholdAction.DOWNGRADE_MODEL

        # Between thresholds
        assert budget.get_current_threshold_action(0.8) == ThresholdAction.DOWNGRADE_MODEL

        # At second threshold
        assert budget.get_current_threshold_action(0.9) == ThresholdAction.DOWNGRADE_MODEL

    def test_hard_limit_exceeded(self):
        budget = BudgetSpec(
            id="test",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(),
            hard_limit=True,
        )

        assert not budget.is_hard_limit_exceeded(0.99)
        assert budget.is_hard_limit_exceeded(1.0)
        assert budget.is_hard_limit_exceeded(1.5)

        # With hard_limit disabled
        budget.hard_limit = False
        assert not budget.is_hard_limit_exceeded(1.5)


class TestPeriodBoundaries:
    """Tests for period boundary calculation."""

    def test_hourly_boundaries(self):
        ref = datetime(2024, 6, 15, 14, 30, 45)
        start, end = get_period_boundaries(BudgetPeriod.HOURLY, ref)

        assert start == datetime(2024, 6, 15, 14, 0, 0)
        assert end == datetime(2024, 6, 15, 15, 0, 0)

    def test_daily_boundaries(self):
        ref = datetime(2024, 6, 15, 14, 30, 45)
        start, end = get_period_boundaries(BudgetPeriod.DAILY, ref)

        assert start == datetime(2024, 6, 15, 0, 0, 0)
        assert end == datetime(2024, 6, 16, 0, 0, 0)

    def test_weekly_boundaries(self):
        # Saturday June 15, 2024
        ref = datetime(2024, 6, 15, 14, 30, 45)
        start, end = get_period_boundaries(BudgetPeriod.WEEKLY, ref)

        # Week starts Monday June 10
        assert start == datetime(2024, 6, 10, 0, 0, 0)
        assert end == datetime(2024, 6, 17, 0, 0, 0)

    def test_monthly_boundaries(self):
        ref = datetime(2024, 6, 15, 14, 30, 45)
        start, end = get_period_boundaries(BudgetPeriod.MONTHLY, ref)

        assert start == datetime(2024, 6, 1, 0, 0, 0)
        assert end == datetime(2024, 7, 1, 0, 0, 0)

    def test_monthly_december(self):
        ref = datetime(2024, 12, 15, 14, 30, 45)
        start, end = get_period_boundaries(BudgetPeriod.MONTHLY, ref)

        assert start == datetime(2024, 12, 1, 0, 0, 0)
        assert end == datetime(2025, 1, 1, 0, 0, 0)


class TestBudgetTracker:
    """Tests for BudgetTracker."""

    def test_register_and_get_run(self):
        tracker = BudgetTracker()
        context = RunContext.create(
            tenant_id="t1",
            strand_id="s1",
            workflow_id="w1",
        )
        run_state = RunState(context=context)

        budget = BudgetSpec(
            id="test",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(tenant_id="t1"),
        )

        tracker.register_run(run_state, [budget])

        retrieved = tracker.get_run_state(context.run_id)
        assert retrieved is not None
        assert retrieved.context.tenant_id == "t1"

    def test_update_run_cost(self):
        tracker = BudgetTracker()
        context = RunContext.create(
            tenant_id="t1",
            strand_id="s1",
            workflow_id="w1",
        )
        run_state = RunState(context=context)
        tracker.register_run(run_state, [])

        tracker.update_run_cost(
            run_id=context.run_id,
            model_name="gpt-4",
            model_cost=0.05,
            input_tokens=100,
            output_tokens=50,
        )

        updated = tracker.get_run_state(context.run_id)
        assert updated.total_cost == 0.05
        assert updated.total_input_tokens == 100
        assert updated.total_output_tokens == 50
        assert updated.model_costs["gpt-4"] == 0.05

    def test_unregister_run_updates_period(self):
        tracker = BudgetTracker()
        context = RunContext.create(
            tenant_id="t1",
            strand_id="s1",
            workflow_id="w1",
        )
        run_state = RunState(context=context)

        budget = BudgetSpec(
            id="test",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(tenant_id="t1"),
            max_cost=100.0,
        )

        tracker.register_run(run_state, [budget])

        # Add some cost
        tracker.update_run_cost(
            run_id=context.run_id,
            model_name="gpt-4",
            model_cost=1.50,
            input_tokens=100,
            output_tokens=50,
        )

        # Unregister
        tracker.unregister_run(context.run_id, [budget])

        # Check period usage was updated
        state = tracker.get_or_create_budget_state(budget, "t1", "s1", "w1")
        assert state.usage.total_cost == 1.50
        assert state.usage.total_runs == 1

    def test_concurrent_runs_tracking(self):
        tracker = BudgetTracker()
        budget = BudgetSpec(
            id="test",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(tenant_id="t1"),
            max_concurrent_runs=10,
        )

        # Start 3 runs
        for i in range(3):
            context = RunContext.create(
                tenant_id="t1",
                strand_id="s1",
                workflow_id="w1",
            )
            run_state = RunState(context=context)
            tracker.register_run(run_state, [budget])

        state = tracker.get_or_create_budget_state(budget, "t1", "s1", "w1")
        assert state.concurrent_runs == 3

    def test_check_budget_limits(self):
        tracker = BudgetTracker()
        budget = BudgetSpec(
            id="test",
            scope=BudgetScope.TENANT,
            match=BudgetMatch(tenant_id="t1"),
            max_cost=10.0,
            hard_limit=True,
        )

        # Get state and manually set usage to exceed limit
        state = tracker.get_or_create_budget_state(budget, "t1", "s1", "w1")
        state.usage.total_cost = 15.0  # Over limit

        exceeded = tracker.check_budget_limits("t1", "s1", "w1", [budget])

        assert len(exceeded) == 1
        assert "hard limit exceeded" in exceeded[0][2].lower()
