"""
Integration tests for CostGuard with OpenAI API.

These tests require:
- OPENAI_API_KEY environment variable
- openai package installed (pip install openai)

Run integration tests:
    pytest tests/integration -v

Run unit tests only (default):
    pytest
"""

import os
import uuid

import pytest
from strands import Agent
from strands.models.openai import OpenAIModel

from strands_costguard import (
    CostGuard,
    CostGuardConfig,
    ModelUsage,
)


@pytest.fixture(autouse=True)
def require_api_key():
    """Ensure OPENAI_API_KEY is set for all tests in this module."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY environment variable must be set to run integration tests")


class MockPolicySource:
    """Policy source for integration tests."""

    def __init__(
        self,
        budgets: list[dict] | None = None,
        routing_policies: list[dict] | None = None,
        pricing: dict | None = None,
    ):
        self._budgets = budgets or []
        self._routing = routing_policies or []
        self._pricing = pricing or self._default_pricing()

    def _default_pricing(self) -> dict:
        return {
            "currency": "USD",
            "fallback_input_per_1k": 1.0,
            "fallback_output_per_1k": 3.0,
            "models": {
                "gpt-4o": {
                    "input_per_1k": 2.50,
                    "output_per_1k": 10.00,
                },
                "gpt-4o-mini": {
                    "input_per_1k": 0.15,
                    "output_per_1k": 0.60,
                },
            },
        }

    def load_budgets(self) -> list[dict]:
        return self._budgets

    def load_routing_policies(self) -> list[dict]:
        return self._routing

    def load_pricing(self) -> dict:
        return self._pricing


class TestIntegrationWithOpenAIAPI:
    """Integration tests using the OpenAI API."""

    def test_simple_agent_with_cost_tracking(self):
        """Test that CostGuard tracks costs for a simple agent interaction."""
        budgets = [
            {
                "id": "test-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 10.0,
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
            enable_budget_enforcement=True,
        )
        guard = CostGuard(config=config)

        run_id = str(uuid.uuid4())
        tenant_id = "integration-test"
        strand_id = "test-agent"
        workflow_id = "test-workflow"

        # Start run
        admission = guard.on_run_start(
            tenant_id=tenant_id,
            strand_id=strand_id,
            workflow_id=workflow_id,
            run_id=run_id,
        )
        assert admission.allowed is True

        # Check model call is allowed
        model_decision = guard.before_model_call(
            run_id=run_id,
            model_name="gpt-4o-mini",
            stage="planning",
            prompt_tokens_estimate=100,
        )
        assert model_decision.allowed is True

        # Create agent and make a real API call
        model = OpenAIModel(model_id="gpt-4o-mini", max_tokens=100)
        agent = Agent(model=model)

        # Simple prompt to minimize token usage
        result = agent("Say 'Hello' and nothing else.")

        # Extract actual token usage from the agent's metrics
        accumulated_usage = result.metrics.accumulated_usage
        prompt_tokens = accumulated_usage.get("inputTokens", 0)
        completion_tokens = accumulated_usage.get("outputTokens", 0)

        # Verify we got real token counts
        assert prompt_tokens > 0, "Expected real input tokens from API"
        assert completion_tokens > 0, "Expected real output tokens from API"

        # Record usage with CostGuard
        usage = ModelUsage.from_response(
            model_name="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        guard.after_model_call(run_id=run_id, usage=usage)

        # Verify cost was tracked (must check before on_run_end as it cleans up run state)
        run_cost = guard.get_run_cost(run_id)
        assert run_cost is not None
        assert run_cost > 0

        # End run
        guard.on_run_end(run_id=run_id, status="completed")

        # Verify budget summary reflects the cost
        summary = guard.get_budget_summary(tenant_id, strand_id, workflow_id)
        assert "test-budget" in summary
        assert summary["test-budget"]["current_cost"] > 0

        guard.shutdown()

    def test_budget_enforcement_blocks_expensive_runs(self):
        """Test that budget enforcement rejects runs when budget is exceeded."""
        budgets = [
            {
                "id": "tight-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 0.001,  # Very low - $0.001
                "hard_limit": True,
                "on_hard_limit_exceeded": "REJECT_NEW_RUNS",
            }
        ]
        config = CostGuardConfig(
            policy_source=MockPolicySource(budgets=budgets),
            enable_metrics=False,
            enable_budget_enforcement=True,
        )
        guard = CostGuard(config=config)

        tenant_id = "integration-test"
        strand_id = "test-agent"
        workflow_id = "test-workflow"

        # First run - should be admitted
        run_id_1 = str(uuid.uuid4())
        admission_1 = guard.on_run_start(
            tenant_id=tenant_id,
            strand_id=strand_id,
            workflow_id=workflow_id,
            run_id=run_id_1,
        )
        assert admission_1.allowed is True

        # Make a real API call
        model = OpenAIModel(model_id="gpt-4o-mini", max_tokens=100)
        agent = Agent(model=model)
        result = agent("Say 'Hi'")

        # Extract actual token usage and record with explicit cost to exceed budget
        accumulated_usage = result.metrics.accumulated_usage
        prompt_tokens = accumulated_usage.get("inputTokens", 0)
        completion_tokens = accumulated_usage.get("outputTokens", 0)

        # Record usage with explicit cost that exceeds the $0.001 budget
        usage = ModelUsage.from_response(
            model_name="gpt-4o-mini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=0.01,  # Explicit cost to exceed $0.001 budget
        )
        guard.after_model_call(run_id=run_id_1, usage=usage)
        guard.on_run_end(run_id=run_id_1, status="completed")

        # Second run - should be rejected due to budget exceeded
        run_id_2 = str(uuid.uuid4())
        admission_2 = guard.on_run_start(
            tenant_id=tenant_id,
            strand_id=strand_id,
            workflow_id=workflow_id,
            run_id=run_id_2,
        )
        assert admission_2.allowed is False
        assert "exceeded" in admission_2.reason.lower()

        guard.shutdown()

    def test_model_routing_with_real_api(self):
        """Test that model routing works with real API calls."""
        routing = [
            {
                "id": "test-routing",
                "match": {"strand_id": "*"},
                "stages": [
                    {
                        "stage": "planning",
                        "default_model": "gpt-4o-mini",
                        "max_tokens": 100,
                    },
                    {
                        "stage": "synthesis",
                        "default_model": "gpt-4o",
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

        run_id = str(uuid.uuid4())

        admission = guard.on_run_start(
            tenant_id="integration-test",
            strand_id="test-agent",
            workflow_id="test-workflow",
            run_id=run_id,
        )
        assert admission.allowed is True

        # Planning stage should route to gpt-4o-mini
        planning_decision = guard.before_model_call(
            run_id=run_id,
            model_name="gpt-4o",  # Request gpt-4o
            stage="planning",
            prompt_tokens_estimate=50,
        )
        assert planning_decision.effective_model == "gpt-4o-mini"
        assert planning_decision.max_tokens == 100

        # Make real API call with the routed model
        model = OpenAIModel(model_id=planning_decision.effective_model, max_tokens=100)
        agent = Agent(model=model)
        result = agent("Say 'Test'")

        # Extract actual token usage from the API response
        accumulated_usage = result.metrics.accumulated_usage
        prompt_tokens = accumulated_usage.get("inputTokens", 0)
        completion_tokens = accumulated_usage.get("outputTokens", 0)

        # Record usage with real token counts
        usage = ModelUsage.from_response(
            model_name=planning_decision.effective_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        guard.after_model_call(run_id=run_id, usage=usage)

        # Synthesis stage should use gpt-4o
        synthesis_decision = guard.before_model_call(
            run_id=run_id,
            model_name="gpt-4o",
            stage="synthesis",
            prompt_tokens_estimate=50,
        )
        assert synthesis_decision.effective_model == "gpt-4o"

        guard.on_run_end(run_id=run_id, status="completed")
        guard.shutdown()
