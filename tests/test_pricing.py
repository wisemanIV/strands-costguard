"""Tests for pricing and cost computation."""

import pytest

from strands_costguard.pricing.table import (
    DEFAULT_MODEL_PRICING,
    ModelPricing,
    PricingTable,
    ToolPricing,
)


class TestModelPricing:
    """Tests for ModelPricing."""

    def test_basic_cost_calculation(self):
        pricing = ModelPricing(
            model_name="test-model",
            input_per_1k=1.0,
            output_per_1k=2.0,
        )

        cost = pricing.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
        )

        # 1000 tokens * $1/1k = $1.00 input
        # 500 tokens * $2/1k = $1.00 output
        assert cost == 2.0

    def test_fractional_tokens(self):
        pricing = ModelPricing(
            model_name="test-model",
            input_per_1k=1.0,
            output_per_1k=2.0,
        )

        cost = pricing.calculate_cost(
            prompt_tokens=250,
            completion_tokens=100,
        )

        # 250/1000 * $1.00 = $0.25
        # 100/1000 * $2.00 = $0.20
        assert cost == pytest.approx(0.45)

    def test_cached_tokens(self):
        pricing = ModelPricing(
            model_name="test-model",
            input_per_1k=1.0,
            output_per_1k=2.0,
            cached_input_per_1k=0.5,  # 50% discount
        )

        cost = pricing.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=500,  # Half of input is cached
        )

        # Standard input: 500 tokens * $1/1k = $0.50
        # Cached input: 500 tokens * $0.5/1k = $0.25
        # Output: 500 tokens * $2/1k = $1.00
        assert cost == pytest.approx(1.75)

    def test_reasoning_tokens(self):
        pricing = ModelPricing(
            model_name="o1-mini",
            input_per_1k=3.0,
            output_per_1k=12.0,
            reasoning_per_1k=12.0,
        )

        cost = pricing.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            reasoning_tokens=2000,
        )

        # Input: 1000 * $3/1k = $3.00
        # Output: 500 * $12/1k = $6.00
        # Reasoning: 2000 * $12/1k = $24.00
        assert cost == pytest.approx(33.0)

    def test_estimate_cost(self):
        pricing = ModelPricing(
            model_name="test-model",
            input_per_1k=1.0,
            output_per_1k=3.0,
        )

        input_estimate = pricing.estimate_cost(2000, is_input=True)
        output_estimate = pricing.estimate_cost(2000, is_input=False)

        assert input_estimate == 2.0
        assert output_estimate == 6.0


class TestToolPricing:
    """Tests for ToolPricing."""

    def test_fixed_cost_per_call(self):
        pricing = ToolPricing(
            tool_name="web_search",
            cost_per_call=0.01,
        )

        cost = pricing.calculate_cost()
        assert cost == 0.01

    def test_data_based_pricing(self):
        pricing = ToolPricing(
            tool_name="file_upload",
            cost_per_call=0.0,
            cost_per_input_byte=0.000001,  # $1 per MB
            cost_per_output_byte=0.0,
        )

        cost = pricing.calculate_cost(input_size_bytes=1_000_000)  # 1 MB
        assert cost == pytest.approx(1.0)

    def test_combined_pricing(self):
        pricing = ToolPricing(
            tool_name="expensive_tool",
            cost_per_call=0.05,
            cost_per_input_byte=0.0000001,
            cost_per_output_byte=0.0000002,
        )

        cost = pricing.calculate_cost(
            input_size_bytes=10000,
            output_size_bytes=5000,
        )

        # $0.05 + (10000 * $0.0000001) + (5000 * $0.0000002)
        # $0.05 + $0.001 + $0.001
        assert cost == pytest.approx(0.052)


class TestPricingTable:
    """Tests for PricingTable."""

    def test_default_pricing_loaded(self):
        table = PricingTable()

        # Should have default models
        assert "gpt-4o" in table.models
        assert "claude-3.5-sonnet" in table.models

    def test_get_known_model(self):
        table = PricingTable()

        pricing = table.get_model_pricing("gpt-4o-mini")
        assert pricing.model_name == "gpt-4o-mini"
        assert pricing.input_per_1k == DEFAULT_MODEL_PRICING["gpt-4o-mini"]["input_per_1k"]

    def test_get_unknown_model_fallback(self):
        table = PricingTable(
            fallback_input_per_1k=5.0,
            fallback_output_per_1k=10.0,
        )

        pricing = table.get_model_pricing("unknown-model-xyz")
        assert pricing.model_name == "unknown-model-xyz"
        assert pricing.input_per_1k == 5.0
        assert pricing.output_per_1k == 10.0

    def test_prefix_matching(self):
        table = PricingTable()

        # Version suffix should match base model
        # Use claude model which has no prefix collision issues
        pricing = table.get_model_pricing("claude-3.5-sonnet-20241022")
        assert pricing.input_per_1k == DEFAULT_MODEL_PRICING["claude-3.5-sonnet"]["input_per_1k"]

    def test_calculate_model_cost(self):
        table = PricingTable()

        cost = table.calculate_model_cost(
            model_name="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=500,
        )

        expected = (1000 / 1000 * 0.15) + (500 / 1000 * 0.60)
        assert cost == pytest.approx(expected)

    def test_from_dict(self):
        data = {
            "currency": "USD",
            "fallback_input_per_1k": 2.0,
            "fallback_output_per_1k": 4.0,
            "models": {
                "custom-model": {
                    "input_per_1k": 1.5,
                    "output_per_1k": 3.0,
                    "cached_input_per_1k": 0.75,
                },
            },
            "tools": {
                "custom-tool": {
                    "cost_per_call": 0.02,
                },
            },
        }

        table = PricingTable.from_dict(data)

        assert table.currency == "USD"
        assert table.fallback_input_per_1k == 2.0
        assert "custom-model" in table.models
        assert table.models["custom-model"].cached_input_per_1k == 0.75
        assert "custom-tool" in table.tools

    def test_tool_pricing_default_zero(self):
        table = PricingTable()

        pricing = table.get_tool_pricing("unknown-tool")
        assert pricing.cost_per_call == 0.0
        cost = pricing.calculate_cost()
        assert cost == 0.0
