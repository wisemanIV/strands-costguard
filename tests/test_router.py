"""Tests for the adaptive model router."""

import time
from unittest.mock import MagicMock

import pytest

from strands_costguard.routing.router import (
    ModelCallContext,
    ModelRouter,
    RouterConfig,
)


class MockCostGuard:
    """Mock CostGuard for testing the router."""

    def __init__(self):
        self.before_model_call_response = MagicMock()
        self.before_model_call_response.allowed = True
        self.before_model_call_response.effective_model = "gpt-4o"
        self.before_model_call_response.max_tokens = None
        self.before_model_call_response.was_downgraded = False
        self.before_model_call_response.reason = None
        self.before_model_call_response.warnings = []

        self.recorded_usage = None

    def before_model_call(self, run_id, model_name, stage, prompt_tokens_estimate):
        return self.before_model_call_response

    def after_model_call(self, run_id, usage):
        self.recorded_usage = usage


class MockModelClient:
    """Mock model client for testing."""

    def __init__(self, response: dict = None):
        self.response = response or {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "choices": [{"message": {"content": "Hello!"}}],
        }
        self.calls = []

    def call(self, messages, model, max_tokens=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        return self.response


class TestRouterConfig:
    """Tests for RouterConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RouterConfig()
        assert config.retry_on_rate_limit is True
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.track_latency is True


class TestModelRouter:
    """Tests for ModelRouter."""

    @pytest.fixture
    def mock_guard(self):
        return MockCostGuard()

    @pytest.fixture
    def router(self, mock_guard):
        return ModelRouter(cost_guard=mock_guard)

    def test_before_call_allowed(self, router, mock_guard):
        """Should return allowed context when CostGuard allows."""
        messages = [{"role": "user", "content": "Hello"}]

        context = router.before_call(
            run_id="run-1",
            stage="planning",
            messages=messages,
            requested_model="gpt-4o",
        )

        assert context.allowed is True
        assert context.effective_model == "gpt-4o"
        assert context.was_downgraded is False

    def test_before_call_rejected(self, router, mock_guard):
        """Should return rejected context when CostGuard rejects."""
        mock_guard.before_model_call_response.allowed = False
        mock_guard.before_model_call_response.reason = "budget exceeded"

        messages = [{"role": "user", "content": "Hello"}]

        context = router.before_call(
            run_id="run-1",
            stage="planning",
            messages=messages,
        )

        assert context.allowed is False
        assert context.reason == "budget exceeded"

    def test_before_call_with_downgrade(self, router, mock_guard):
        """Should capture downgrade information."""
        mock_guard.before_model_call_response.was_downgraded = True
        mock_guard.before_model_call_response.effective_model = "gpt-4o-mini"

        messages = [{"role": "user", "content": "Hello"}]

        context = router.before_call(
            run_id="run-1",
            stage="synthesis",
            messages=messages,
            requested_model="gpt-4o",
        )

        assert context.was_downgraded is True
        assert context.effective_model == "gpt-4o-mini"
        assert context.requested_model == "gpt-4o"

    def test_before_call_with_max_tokens(self, router, mock_guard):
        """Should pass through max_tokens from CostGuard."""
        mock_guard.before_model_call_response.max_tokens = 1000

        messages = [{"role": "user", "content": "Hello"}]

        context = router.before_call(
            run_id="run-1",
            stage="planning",
            messages=messages,
        )

        assert context.max_tokens == 1000

    def test_before_call_tracks_pending(self, router, mock_guard):
        """Should track pending calls for latency calculation."""
        messages = [{"role": "user", "content": "Hello"}]

        context = router.before_call(
            run_id="run-1",
            stage="planning",
            messages=messages,
        )

        assert "run-1" in router._pending_calls
        assert router._pending_calls["run-1"]["context"] == context

    def test_before_call_no_pending_when_rejected(self, router, mock_guard):
        """Should not track pending calls when rejected."""
        mock_guard.before_model_call_response.allowed = False

        messages = [{"role": "user", "content": "Hello"}]

        router.before_call(
            run_id="run-1",
            stage="planning",
            messages=messages,
        )

        assert "run-1" not in router._pending_calls

    def test_after_call_records_usage(self, router, mock_guard):
        """Should record usage with CostGuard."""
        # First, do before_call to set up pending
        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        response = {
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
        }

        usage = router.after_call(run_id="run-1", response=response)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert mock_guard.recorded_usage is not None

    def test_after_call_extracts_cached_tokens(self, router, mock_guard):
        """Should extract cached tokens from response."""
        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cached_tokens": 30,
            },
        }

        usage = router.after_call(run_id="run-1", response=response)

        assert usage.cached_tokens == 30

    def test_after_call_extracts_reasoning_tokens(self, router, mock_guard):
        """Should extract reasoning tokens from response."""
        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 20,
            },
        }

        usage = router.after_call(run_id="run-1", response=response)

        assert usage.reasoning_tokens == 20

    def test_after_call_calculates_latency(self, router, mock_guard):
        """Should calculate latency when tracking enabled."""
        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        # Simulate some time passing
        time.sleep(0.01)  # 10ms

        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        usage = router.after_call(run_id="run-1", response=response)

        assert usage.latency_ms is not None
        assert usage.latency_ms >= 10  # At least 10ms

    def test_after_call_no_latency_when_disabled(self, mock_guard):
        """Should not calculate latency when tracking disabled."""
        config = RouterConfig(track_latency=False)
        router = ModelRouter(cost_guard=mock_guard, config=config)

        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        usage = router.after_call(run_id="run-1", response=response)

        assert usage.latency_ms is None

    def test_after_call_uses_model_from_context(self, router, mock_guard):
        """Should use model name from context when available."""
        mock_guard.before_model_call_response.effective_model = "gpt-4o-mini"

        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        usage = router.after_call(run_id="run-1", response=response)

        assert usage.model_name == "gpt-4o-mini"

    def test_after_call_uses_explicit_model_name(self, router, mock_guard):
        """Should use explicit model_name when provided."""
        messages = [{"role": "user", "content": "Hello"}]
        router.before_call(run_id="run-1", stage="planning", messages=messages)

        response = {"model": "gpt-4o", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}

        usage = router.after_call(run_id="run-1", response=response, model_name="custom-model")

        assert usage.model_name == "custom-model"

    def test_after_call_without_pending(self, router, mock_guard):
        """Should handle after_call without prior before_call."""
        response = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        usage = router.after_call(run_id="run-unknown", response=response)

        # Should still work, using model from response
        assert usage.model_name == "gpt-4o"
        assert usage.latency_ms is None

    def test_call_makes_model_call(self, router, mock_guard):
        """Should make complete model call through client."""
        client = MockModelClient()
        messages = [{"role": "user", "content": "Hello"}]

        response, usage = router.call(
            run_id="run-1",
            stage="planning",
            messages=messages,
            client=client,
            requested_model="gpt-4o",
        )

        assert response["choices"][0]["message"]["content"] == "Hello!"
        assert len(client.calls) == 1
        assert client.calls[0]["model"] == "gpt-4o"
        assert usage.prompt_tokens == 100

    def test_call_uses_effective_model(self, router, mock_guard):
        """Should use effective model from CostGuard decision."""
        mock_guard.before_model_call_response.effective_model = "gpt-4o-mini"
        mock_guard.before_model_call_response.was_downgraded = True

        client = MockModelClient()
        messages = [{"role": "user", "content": "Hello"}]

        router.call(
            run_id="run-1",
            stage="synthesis",
            messages=messages,
            client=client,
            requested_model="gpt-4o",
        )

        assert client.calls[0]["model"] == "gpt-4o-mini"

    def test_call_passes_max_tokens(self, router, mock_guard):
        """Should pass max_tokens from CostGuard to client."""
        mock_guard.before_model_call_response.max_tokens = 500

        client = MockModelClient()
        messages = [{"role": "user", "content": "Hello"}]

        router.call(
            run_id="run-1",
            stage="planning",
            messages=messages,
            client=client,
        )

        assert client.calls[0]["max_tokens"] == 500

    def test_call_raises_when_rejected(self, router, mock_guard):
        """Should raise RuntimeError when call is rejected."""
        mock_guard.before_model_call_response.allowed = False
        mock_guard.before_model_call_response.reason = "budget exceeded"

        client = MockModelClient()
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(RuntimeError) as exc_info:
            router.call(
                run_id="run-1",
                stage="planning",
                messages=messages,
                client=client,
            )

        assert "budget exceeded" in str(exc_info.value)

    def test_call_passes_kwargs(self, router, mock_guard):
        """Should pass additional kwargs to client."""
        client = MockModelClient()
        messages = [{"role": "user", "content": "Hello"}]

        router.call(
            run_id="run-1",
            stage="planning",
            messages=messages,
            client=client,
            temperature=0.7,
            top_p=0.9,
        )

        assert client.calls[0]["temperature"] == 0.7
        assert client.calls[0]["top_p"] == 0.9


class TestTokenEstimation:
    """Tests for token estimation."""

    @pytest.fixture
    def router(self):
        return ModelRouter(cost_guard=MockCostGuard())

    def test_estimate_tokens_simple(self, router):
        """Should estimate tokens for simple messages."""
        messages = [
            {"role": "user", "content": "Hello, world!"},  # 13 chars + 10 overhead = 23 chars
        ]

        estimate = router._estimate_tokens(messages)

        # ~4 chars per token: 23 // 4 = 5
        assert estimate > 0
        assert estimate < 100  # Reasonable for short message

    def test_estimate_tokens_multiple_messages(self, router):
        """Should sum tokens across multiple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        estimate = router._estimate_tokens(messages)

        assert estimate > 10  # Should be reasonable for multiple messages

    def test_estimate_tokens_multipart_content(self, router):
        """Should handle multipart content (e.g., vision)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]

        estimate = router._estimate_tokens(messages)

        assert estimate > 0

    def test_estimate_tokens_empty_content(self, router):
        """Should handle empty content."""
        messages = [{"role": "user", "content": ""}]

        estimate = router._estimate_tokens(messages)

        # Should have some overhead even for empty content
        assert estimate >= 0


class TestModelCallContext:
    """Tests for ModelCallContext."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        context = ModelCallContext(
            run_id="run-1",
            stage="planning",
            requested_model="gpt-4o",
            effective_model="gpt-4o-mini",
            max_tokens=1000,
            allowed=True,
            was_downgraded=True,
            reason="soft threshold exceeded",
            warnings=["approaching budget limit"],
            prompt_tokens_estimate=500,
        )

        result = context.to_dict()

        assert result["run_id"] == "run-1"
        assert result["stage"] == "planning"
        assert result["requested_model"] == "gpt-4o"
        assert result["effective_model"] == "gpt-4o-mini"
        assert result["max_tokens"] == 1000
        assert result["allowed"] is True
        assert result["was_downgraded"] is True
        assert result["reason"] == "soft threshold exceeded"
        assert result["warnings"] == ["approaching budget limit"]
        assert result["prompt_tokens_estimate"] == 500
