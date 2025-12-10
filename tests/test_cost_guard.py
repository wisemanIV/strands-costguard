"""Tests for the main CostGuard runtime."""

from strands_costguard.core.config import CostGuardConfig
from strands_costguard.core.cost_guard import CostGuard
from strands_costguard.core.usage import ModelUsage, ToolUsage


class MockPolicySource:
    """Mock policy source for testing."""

    def __init__(
        self,
        budgets: list[dict] = None,
        routing_policies: list[dict] = None,
        pricing: dict = None,
    ):
        self._budgets = budgets or []
        self._routing = routing_policies or []
        self._pricing = pricing or {}

    def load_budgets(self) -> list[dict]:
        return self._budgets

    def load_routing_policies(self) -> list[dict]:
        return self._routing

    def load_pricing(self) -> dict:
        return self._pricing


class TestCostGuardLifecycle:
    """Tests for Cost Guard lifecycle hooks."""

    def test_run_admission_no_budgets(self):
        """Run should be admitted when no budgets are configured."""
        config = CostGuardConfig(
            policy_source=MockPolicySource(),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        decision = guard.on_run_start(
            tenant_id="t1",
            strand_id="s1",
            workflow_id="w1",
            run_id="run-1",
        )

        assert decision.allowed is True

    def test_run_admission_with_budget(self):
        """Run should be admitted when under budget."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 100.0,
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        decision = guard.on_run_start(
            tenant_id="t1",
            strand_id="s1",
            workflow_id="w1",
            run_id="run-1",
        )

        assert decision.allowed is True
        assert decision.remaining_budget == 100.0
        assert decision.budget_utilization == 0.0

    def test_run_rejection_budget_exceeded(self):
        """Run should be rejected when budget is exceeded."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 100.0,
                "hard_limit": True,
                "on_hard_limit_exceeded": "REJECT_NEW_RUNS",
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        # Start first run and add cost to exceed budget
        guard.on_run_start("t1", "s1", "w1", "run-1")
        guard.after_model_call(
            "run-1",
            ModelUsage.from_response("gpt-4", 50000, 50000, cost=150.0),
        )
        guard.on_run_end("run-1", "completed")

        # Second run should be rejected
        decision = guard.on_run_start("t1", "s1", "w1", "run-2")
        assert decision.allowed is False
        assert "exceeded" in decision.reason.lower()

    def test_iteration_proceeds_under_limit(self):
        """Iteration should proceed when under limits."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "constraints": {"max_iterations_per_run": 10},
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        decision = guard.before_iteration("run-1", iteration_idx=0)
        assert decision.allowed is True
        assert decision.remaining_iterations == 10

    def test_iteration_halted_at_limit(self):
        """Iteration should be halted when max iterations reached."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "constraints": {"max_iterations_per_run": 5},
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        # Should allow up to iteration 4 (0-indexed)
        for i in range(5):
            decision = guard.before_iteration("run-1", iteration_idx=i)
            assert decision.allowed is True

        # Iteration 5 should be halted
        decision = guard.before_iteration("run-1", iteration_idx=5)
        assert decision.allowed is False
        assert "max iterations" in decision.reason.lower()

    def test_model_call_allowed(self):
        """Model call should be allowed under normal conditions."""
        config = CostGuardConfig(
            policy_source=MockPolicySource(),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        decision = guard.before_model_call(
            run_id="run-1",
            model_name="gpt-4o",
            stage="planning",
            prompt_tokens_estimate=500,
        )

        assert decision.allowed is True
        assert decision.effective_model == "gpt-4o"

    def test_model_call_records_cost(self):
        """Model call should record cost correctly."""
        config = CostGuardConfig(
            policy_source=MockPolicySource(),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        usage = ModelUsage.from_response(
            model_name="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        guard.after_model_call("run-1", usage)

        run_cost = guard.get_run_cost("run-1")
        assert run_cost > 0

    def test_tool_call_allowed(self):
        """Tool call should be allowed under normal conditions."""
        config = CostGuardConfig(
            policy_source=MockPolicySource(),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        decision = guard.before_tool_call("run-1", "web_search")

        assert decision.allowed is True

    def test_tool_call_rejected_at_limit(self):
        """Tool call should be rejected when max calls reached."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "constraints": {"max_tool_calls_per_run": 3},
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        # Make 3 tool calls
        for i in range(3):
            decision = guard.before_tool_call("run-1", f"tool_{i}")
            assert decision.allowed is True
            guard.after_tool_call("run-1", f"tool_{i}", ToolUsage(tool_name=f"tool_{i}"))

        # 4th tool call should be rejected
        decision = guard.before_tool_call("run-1", "tool_4")
        assert decision.allowed is False
        assert "max tool calls" in decision.reason.lower()

    def test_budget_summary(self):
        """Budget summary should return correct information."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "period": "monthly",
                "max_cost": 100.0,
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")
        guard.after_model_call(
            "run-1",
            ModelUsage.from_response("gpt-4o", 1000, 500, cost=5.0),
        )
        guard.on_run_end("run-1", "completed")

        summary = guard.get_budget_summary("t1", "s1", "w1")

        assert "test-budget" in summary
        assert summary["test-budget"]["current_cost"] == 5.0
        assert summary["test-budget"]["utilization"] == 0.05
        assert summary["test-budget"]["remaining"] == 95.0


class TestCostGuardRouting:
    """Tests for Cost Guard model routing."""

    def test_routing_with_policy(self):
        """Model should be routed according to policy."""
        routing = [
            {
                "id": "test-routing",
                "match": {"strand_id": "*"},
                "stages": [
                    {
                        "stage": "planning",
                        "default_model": "gpt-4o-mini",
                        "max_tokens": 2000,
                    },
                    {
                        "stage": "synthesis",
                        "default_model": "gpt-4o",
                        "fallback_model": "gpt-4o-mini",
                        "trigger_downgrade_on": {"soft_threshold_exceeded": True},
                    },
                ],
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(routing_policies=routing),
            enable_metrics=False,
            enable_routing=True,
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        # Planning stage should use gpt-4o-mini
        decision = guard.before_model_call("run-1", "gpt-4o", "planning", 500)
        assert decision.effective_model == "gpt-4o-mini"
        assert decision.max_tokens == 2000

        # Synthesis stage should use gpt-4o
        decision = guard.before_model_call("run-1", "gpt-4o", "synthesis", 500)
        assert decision.effective_model == "gpt-4o"

    def test_model_downgrade_on_threshold(self):
        """Model should be downgraded when soft threshold exceeded."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 10.0,
                "soft_thresholds": [0.7],
                "on_soft_threshold_exceeded": "DOWNGRADE_MODEL",
            }
        ]
        routing = [
            {
                "id": "test-routing",
                "match": {"strand_id": "*"},
                "stages": [
                    {
                        "stage": "synthesis",
                        "default_model": "gpt-4o",
                        "fallback_model": "gpt-4o-mini",
                        "trigger_downgrade_on": {"soft_threshold_exceeded": True},
                    },
                ],
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets, routing_policies=routing),
            enable_metrics=False,
            enable_routing=True,
        )
        guard = CostGuard(config=config)

        # First run: add cost to exceed 70% threshold
        guard.on_run_start("t1", "s1", "w1", "run-1")
        guard.after_model_call(
            "run-1",
            ModelUsage.from_response("gpt-4o", 1000, 500, cost=8.0),
        )
        # End the run to commit costs to budget state
        guard.on_run_end("run-1", "completed")

        # Second run: should be downgraded since budget utilization is 80%
        guard.on_run_start("t1", "s1", "w1", "run-2")
        decision = guard.before_model_call("run-2", "gpt-4o", "synthesis", 500)
        assert decision.was_downgraded is True
        assert decision.effective_model == "gpt-4o-mini"


class TestCostGuardDisabled:
    """Tests for disabled Cost Guard features."""

    def test_budget_enforcement_disabled(self):
        """Budget enforcement can be disabled."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 1.0,  # Very low limit
                "hard_limit": True,
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
            enable_budget_enforcement=False,  # Disabled
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")
        guard.after_model_call(
            "run-1",
            ModelUsage.from_response("gpt-4o", 10000, 5000, cost=100.0),
        )
        guard.on_run_end("run-1", "completed")

        # Should still admit new runs despite exceeding budget
        decision = guard.on_run_start("t1", "s1", "w1", "run-2")
        assert decision.allowed is True

    def test_routing_disabled(self):
        """Routing can be disabled."""
        routing = [
            {
                "id": "test-routing",
                "match": {"strand_id": "*"},
                "stages": [
                    {"stage": "planning", "default_model": "gpt-4o-mini"},
                ],
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(routing_policies=routing),
            enable_metrics=False,
            enable_routing=False,  # Disabled
        )
        guard = CostGuard(config=config)

        guard.on_run_start("t1", "s1", "w1", "run-1")

        # Should use requested model, not routing policy
        decision = guard.before_model_call("run-1", "gpt-4o", "planning", 500)
        assert decision.effective_model == "gpt-4o"
