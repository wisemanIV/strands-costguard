"""Tests for routing policies and model router."""

from strands_costguard.policies.routing import (
    DowngradeTrigger,
    RoutingPolicy,
    StageConfig,
)
from strands_costguard.routing.router import ModelCallContext


class TestDowngradeTrigger:
    """Tests for DowngradeTrigger."""

    def test_no_trigger(self):
        trigger = DowngradeTrigger()
        should, reason = trigger.should_downgrade()
        assert should is False
        assert reason == ""

    def test_soft_threshold_trigger(self):
        trigger = DowngradeTrigger(soft_threshold_exceeded=True)

        # Should not trigger when threshold not exceeded
        should, _ = trigger.should_downgrade(soft_threshold_exceeded=False)
        assert should is False

        # Should trigger when threshold exceeded
        should, reason = trigger.should_downgrade(soft_threshold_exceeded=True)
        assert should is True
        assert "threshold" in reason.lower()

    def test_remaining_budget_trigger(self):
        trigger = DowngradeTrigger(remaining_budget_below=10.0)

        # Should not trigger when budget is sufficient
        should, _ = trigger.should_downgrade(remaining_budget=15.0)
        assert should is False

        # Should trigger when budget is low
        should, reason = trigger.should_downgrade(remaining_budget=5.0)
        assert should is True
        assert "budget" in reason.lower()

    def test_iteration_count_trigger(self):
        trigger = DowngradeTrigger(iteration_count_above=5)

        # Should not trigger when iterations are low
        should, _ = trigger.should_downgrade(iteration_count=3)
        assert should is False

        # Should trigger when iterations are high
        should, reason = trigger.should_downgrade(iteration_count=6)
        assert should is True
        assert "iteration" in reason.lower()

    def test_latency_trigger(self):
        trigger = DowngradeTrigger(latency_above_ms=1000.0)

        # Should not trigger when latency is low
        should, _ = trigger.should_downgrade(avg_latency_ms=500.0)
        assert should is False

        # Should trigger when latency is high
        should, reason = trigger.should_downgrade(avg_latency_ms=1500.0)
        assert should is True
        assert "latency" in reason.lower()

    def test_multiple_triggers(self):
        trigger = DowngradeTrigger(
            soft_threshold_exceeded=True,
            remaining_budget_below=10.0,
        )

        # Should trigger on first matching condition
        should, reason = trigger.should_downgrade(
            soft_threshold_exceeded=True,
            remaining_budget=15.0,  # Above threshold
        )
        assert should is True
        assert "threshold" in reason.lower()

    def test_from_dict(self):
        data = {
            "soft_threshold_exceeded": True,
            "remaining_budget_below": 5.0,
            "iteration_count_above": 10,
        }
        trigger = DowngradeTrigger.from_dict(data)

        assert trigger.soft_threshold_exceeded is True
        assert trigger.remaining_budget_below == 5.0
        assert trigger.iteration_count_above == 10


class TestStageConfig:
    """Tests for StageConfig."""

    def test_basic_model_selection(self):
        config = StageConfig(
            stage="planning",
            default_model="gpt-4o-mini",
        )

        model, downgraded, reason = config.get_effective_model()
        assert model == "gpt-4o-mini"
        assert downgraded is False
        assert reason == ""

    def test_fallback_without_trigger(self):
        config = StageConfig(
            stage="synthesis",
            default_model="gpt-4o",
            fallback_model="gpt-4o-mini",
            trigger_downgrade_on=DowngradeTrigger(),  # No triggers set
        )

        model, downgraded, _ = config.get_effective_model()
        assert model == "gpt-4o"
        assert downgraded is False

    def test_fallback_with_trigger(self):
        config = StageConfig(
            stage="synthesis",
            default_model="gpt-4o",
            fallback_model="gpt-4o-mini",
            trigger_downgrade_on=DowngradeTrigger(soft_threshold_exceeded=True),
        )

        model, downgraded, reason = config.get_effective_model(soft_threshold_exceeded=True)
        assert model == "gpt-4o-mini"
        assert downgraded is True
        assert reason != ""

    def test_from_dict(self):
        data = {
            "stage": "planning",
            "default_model": "gpt-4o-mini",
            "fallback_model": "gpt-3.5-turbo",
            "max_tokens": 2000,
            "trigger_downgrade_on": {
                "remaining_budget_below": 5.0,
            },
        }
        config = StageConfig.from_dict(data)

        assert config.stage == "planning"
        assert config.default_model == "gpt-4o-mini"
        assert config.fallback_model == "gpt-3.5-turbo"
        assert config.max_tokens == 2000
        assert config.trigger_downgrade_on.remaining_budget_below == 5.0


class TestRoutingPolicy:
    """Tests for RoutingPolicy."""

    def test_wildcard_match(self):
        policy = RoutingPolicy(
            id="default",
            match={"strand_id": "*"},
        )

        assert policy.matches_context("any-tenant", "any-strand", "any-workflow")

    def test_specific_match(self):
        policy = RoutingPolicy(
            id="specific",
            match={
                "tenant_id": "prod",
                "strand_id": "analytics",
            },
        )

        assert policy.matches_context("prod", "analytics", "any-workflow")
        assert not policy.matches_context("dev", "analytics", "any-workflow")
        assert not policy.matches_context("prod", "codegen", "any-workflow")

    def test_disabled_policy(self):
        policy = RoutingPolicy(
            id="disabled",
            match={"strand_id": "*"},
            enabled=False,
        )

        assert not policy.matches_context("any", "any", "any")

    def test_get_stage_config(self):
        policy = RoutingPolicy(
            id="test",
            stages=[
                StageConfig(stage="planning", default_model="gpt-4o-mini"),
                StageConfig(stage="synthesis", default_model="gpt-4o"),
            ],
        )

        planning = policy.get_stage_config("planning")
        assert planning is not None
        assert planning.default_model == "gpt-4o-mini"

        synthesis = policy.get_stage_config("synthesis")
        assert synthesis is not None
        assert synthesis.default_model == "gpt-4o"

        unknown = policy.get_stage_config("unknown")
        assert unknown is None

    def test_get_model_for_stage(self):
        policy = RoutingPolicy(
            id="test",
            default_model="gpt-3.5-turbo",
            stages=[
                StageConfig(
                    stage="planning",
                    default_model="gpt-4o-mini",
                    max_tokens=2000,
                ),
            ],
        )

        # Known stage
        model, max_tokens, downgraded, _ = policy.get_model_for_stage("planning")
        assert model == "gpt-4o-mini"
        assert max_tokens == 2000
        assert downgraded is False

        # Unknown stage falls back to default
        model, max_tokens, downgraded, _ = policy.get_model_for_stage("unknown")
        assert model == "gpt-3.5-turbo"
        assert max_tokens is None
        assert downgraded is False

    def test_specificity_score(self):
        policies = [
            RoutingPolicy(id="global", match={}),
            RoutingPolicy(id="tenant", match={"tenant_id": "t1"}),
            RoutingPolicy(id="strand", match={"strand_id": "s1"}),
            RoutingPolicy(id="workflow", match={"workflow_id": "w1"}),
            RoutingPolicy(
                id="full", match={"tenant_id": "t1", "strand_id": "s1", "workflow_id": "w1"}
            ),
        ]

        scores = [p.specificity_score() for p in policies]
        assert scores == [0, 1, 2, 4, 7]

    def test_from_dict(self):
        data = {
            "id": "test-policy",
            "match": {"strand_id": "analytics"},
            "default_model": "gpt-4o-mini",
            "stages": [
                {
                    "stage": "synthesis",
                    "default_model": "gpt-4o",
                    "fallback_model": "gpt-4o-mini",
                }
            ],
        }
        policy = RoutingPolicy.from_dict(data)

        assert policy.id == "test-policy"
        assert policy.match["strand_id"] == "analytics"
        assert policy.default_model == "gpt-4o-mini"
        assert len(policy.stages) == 1
        assert policy.stages[0].stage == "synthesis"


class TestModelCallContext:
    """Tests for ModelCallContext."""

    def test_to_dict(self):
        context = ModelCallContext(
            run_id="run-1",
            stage="planning",
            requested_model="gpt-4o",
            effective_model="gpt-4o-mini",
            max_tokens=2000,
            allowed=True,
            was_downgraded=True,
            reason="threshold exceeded",
            warnings=["warning 1"],
            prompt_tokens_estimate=500,
        )

        d = context.to_dict()
        assert d["run_id"] == "run-1"
        assert d["stage"] == "planning"
        assert d["requested_model"] == "gpt-4o"
        assert d["effective_model"] == "gpt-4o-mini"
        assert d["was_downgraded"] is True
