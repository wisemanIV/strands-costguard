"""Tests for OpenTelemetry metrics emission."""

from unittest.mock import MagicMock, patch

import pytest

from strands_costguard.core.entities import RunContext, RunState
from strands_costguard.core.usage import IterationUsage, ModelUsage, ToolUsage
from strands_costguard.metrics.otel import (
    MetricsEmitter,
)


@pytest.fixture
def mock_meter():
    """Create a mock meter with mock instruments."""
    meter = MagicMock()
    meter.create_counter.return_value = MagicMock()
    return meter


@pytest.fixture
def run_context():
    """Create a sample run context."""
    return RunContext.create(
        tenant_id="tenant-1",
        strand_id="strand-1",
        workflow_id="workflow-1",
        run_id="run-123",
    )


@pytest.fixture
def run_state(run_context):
    """Create a sample run state."""
    state = RunState(context=run_context)
    state.status = "completed"
    return state


class TestMetricsEmitter:
    """Tests for MetricsEmitter."""

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_initialization_creates_instruments(self, mock_get_meter, mock_meter):
        """Should create metric instruments on initialization."""
        mock_get_meter.return_value = mock_meter

        _emitter = MetricsEmitter()

        mock_get_meter.assert_called_once_with(name="strand-cost-guard", version="0.1.0")
        assert mock_meter.create_counter.call_count >= 5  # Multiple counters created

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_run_start(self, mock_get_meter, mock_meter, run_context):
        """Should record run start event."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter()
        emitter.record_run_start(run_context)

        mock_counter.add.assert_called()
        call_args = mock_counter.add.call_args
        assert call_args[0][0] == 1  # Count of 1
        assert "strands.event" in call_args[1]["attributes"]
        assert call_args[1]["attributes"]["strands.event"] == "start"

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_run_end(self, mock_get_meter, mock_meter, run_state):
        """Should record run end event with costs."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        run_state.add_model_cost("gpt-4o", 5.0, 1000, 500)

        emitter = MetricsEmitter()
        emitter.record_run_end(run_state)

        # Should record run count, cost, and tokens
        assert mock_counter.add.call_count >= 1

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_model_cost(self, mock_get_meter, mock_meter, run_context):
        """Should record model cost and token usage."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        usage = ModelUsage.from_response(
            model_name="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
            cost=2.5,
        )

        emitter = MetricsEmitter()
        emitter.record_model_cost(run_context, usage)

        # Should record cost and tokens
        assert mock_counter.add.call_count >= 1

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_tool_cost(self, mock_get_meter, mock_meter, run_context):
        """Should record tool cost."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        usage = ToolUsage(tool_name="web_search", cost=0.01)

        emitter = MetricsEmitter()
        emitter.record_tool_cost(run_context, usage)

        # Should record tool call count and cost
        assert mock_counter.add.call_count >= 1

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_iteration(self, mock_get_meter, mock_meter, run_context):
        """Should record iteration."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        usage = IterationUsage(iteration_idx=0)

        emitter = MetricsEmitter()
        emitter.record_iteration(run_context, usage)

        mock_counter.add.assert_called()
        call_args = mock_counter.add.call_args
        assert call_args[0][0] == 1
        assert "strands.iteration_idx" in call_args[1]["attributes"]

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_downgrade(self, mock_get_meter, mock_meter, run_context):
        """Should record model downgrade event."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter()
        emitter.record_downgrade(
            run_context,
            original_model="gpt-4o",
            fallback_model="gpt-4o-mini",
            reason="soft threshold exceeded",
        )

        mock_counter.add.assert_called()
        call_args = mock_counter.add.call_args
        assert call_args[0][0] == 1
        attrs = call_args[1]["attributes"]
        assert attrs["genai.model.original"] == "gpt-4o"
        assert attrs["genai.model.fallback"] == "gpt-4o-mini"

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_rejection(self, mock_get_meter, mock_meter, run_context):
        """Should record run rejection event."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter()
        emitter.record_rejection(run_context, reason="budget exceeded")

        mock_counter.add.assert_called()
        call_args = mock_counter.add.call_args
        assert call_args[1]["attributes"]["strands.reason"] == "budget exceeded"

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_record_iteration_halt(self, mock_get_meter, mock_meter, run_context):
        """Should record iteration halt event."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter()
        emitter.record_iteration_halt(run_context, reason="max iterations reached")

        mock_counter.add.assert_called()
        call_args = mock_counter.add.call_args
        assert "max iterations" in call_args[1]["attributes"]["strands.reason"]

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_exclude_run_id_by_default(self, mock_get_meter, mock_meter, run_context):
        """Should exclude run_id from attributes by default (high cardinality)."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter(include_run_id=False)
        emitter.record_run_start(run_context)

        call_args = mock_counter.add.call_args
        attrs = call_args[1]["attributes"]
        assert "strands.run_id" not in attrs

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_include_run_id_when_enabled(self, mock_get_meter, mock_meter, run_context):
        """Should include run_id when explicitly enabled."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter(include_run_id=True)
        emitter.record_run_start(run_context)

        call_args = mock_counter.add.call_args
        attrs = call_args[1]["attributes"]
        assert "strands.run_id" in attrs
        assert attrs["strands.run_id"] == "run-123"

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_reason_truncation(self, mock_get_meter, mock_meter, run_context):
        """Should truncate long reasons to avoid attribute limits."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        long_reason = "x" * 200  # More than 100 chars

        emitter = MetricsEmitter()
        emitter.record_rejection(run_context, reason=long_reason)

        call_args = mock_counter.add.call_args
        attrs = call_args[1]["attributes"]
        assert len(attrs["strands.reason"]) == 100

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_handles_meter_exception(self, mock_get_meter):
        """Should handle exceptions during meter setup gracefully."""
        mock_get_meter.side_effect = Exception("Meter not available")

        # Should not raise, just log warning
        emitter = MetricsEmitter()
        assert emitter._meter is None

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_noop_when_instruments_not_initialized(self, mock_get_meter, run_context):
        """Should be no-op when instruments are not initialized."""
        mock_get_meter.side_effect = Exception("Meter not available")

        emitter = MetricsEmitter()
        # These should not raise
        emitter.record_run_start(run_context)
        emitter.record_downgrade(run_context, "gpt-4o", "gpt-4o-mini", "test")
        emitter.record_rejection(run_context, "test")
        emitter.record_iteration_halt(run_context, "test")

    @patch("strands_costguard.metrics.otel.metrics.get_meter")
    def test_base_attributes_include_context_fields(self, mock_get_meter, mock_meter, run_context):
        """Should include tenant, strand, workflow in base attributes."""
        mock_get_meter.return_value = mock_meter
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        emitter = MetricsEmitter()
        emitter.record_run_start(run_context)

        call_args = mock_counter.add.call_args
        attrs = call_args[1]["attributes"]
        assert "strands.tenant_id" in attrs
        assert "strands.strand_id" in attrs
        assert "strands.workflow_id" in attrs
