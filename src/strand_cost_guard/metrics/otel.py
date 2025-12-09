"""OpenTelemetry metrics emission for Cost Guard."""

import logging
from dataclasses import dataclass
from typing import Optional

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as OTLPMetricExporterHTTP,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from strand_cost_guard.core.config import OtelConfig
from strand_cost_guard.core.entities import RunContext, RunState
from strand_cost_guard.core.usage import IterationUsage, ModelUsage, ToolUsage

logger = logging.getLogger(__name__)

# Metric names following OpenTelemetry semantic conventions
METRIC_COST_TOTAL = "genai.cost.total"
METRIC_COST_MODEL = "genai.cost.model"
METRIC_COST_TOOL = "genai.cost.tool"
METRIC_TOKENS_INPUT = "genai.tokens.input"
METRIC_TOKENS_OUTPUT = "genai.tokens.output"
METRIC_AGENT_ITERATIONS = "genai.agent.iterations"
METRIC_AGENT_TOOL_CALLS = "genai.agent.tool_calls"
METRIC_AGENT_RUNS = "genai.agent.runs"
METRIC_COST_DOWNGRADE_EVENTS = "genai.cost.downgrade_events"
METRIC_COST_REJECTION_EVENTS = "genai.cost.rejection_events"
METRIC_COST_HALT_EVENTS = "genai.cost.halt_events"


@dataclass
class MetricsEmitter:
    """
    OpenTelemetry metrics emitter for Cost Guard.

    Emits cost and usage metrics using the OpenTelemetry SDK, configured
    to export via OTLP to an OTel Collector.
    """

    config: OtelConfig
    _meter_provider: Optional[MeterProvider] = None
    _meter: Optional[metrics.Meter] = None

    # Counters
    _cost_total: Optional[metrics.Counter] = None
    _cost_model: Optional[metrics.Counter] = None
    _cost_tool: Optional[metrics.Counter] = None
    _tokens_input: Optional[metrics.Counter] = None
    _tokens_output: Optional[metrics.Counter] = None
    _iterations: Optional[metrics.Counter] = None
    _tool_calls: Optional[metrics.Counter] = None
    _runs: Optional[metrics.Counter] = None
    _downgrades: Optional[metrics.Counter] = None
    _rejections: Optional[metrics.Counter] = None
    _halts: Optional[metrics.Counter] = None

    def __post_init__(self) -> None:
        """Initialize OpenTelemetry meter and instruments."""
        if not self.config.enabled:
            return

        try:
            self._setup_meter_provider()
            self._setup_instruments()
            logger.info(
                f"OpenTelemetry metrics initialized, exporting to {self.config.endpoint}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry metrics: {e}")

    def _setup_meter_provider(self) -> None:
        """Set up the OpenTelemetry meter provider with OTLP exporter."""
        # Create resource with attributes
        resource = Resource.create(self.config.get_resource_attributes())

        # Create exporter based on configuration
        if self.config.use_grpc:
            exporter = OTLPMetricExporter(
                endpoint=self.config.endpoint,
                timeout=self.config.export_timeout_ms / 1000,
            )
        else:
            exporter = OTLPMetricExporterHTTP(
                endpoint=self.config.endpoint,
                timeout=self.config.export_timeout_ms / 1000,
            )

        # Create metric reader with periodic export
        reader = PeriodicExportingMetricReader(
            exporter=exporter,
            export_interval_millis=self.config.export_interval_ms,
        )

        # Create and set meter provider
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
        )
        metrics.set_meter_provider(self._meter_provider)

        # Get meter
        self._meter = metrics.get_meter(
            name="strand-cost-guard",
            version="0.1.0",
        )

    def _setup_instruments(self) -> None:
        """Create metric instruments."""
        if not self._meter:
            return

        # Cost counters
        self._cost_total = self._meter.create_counter(
            name=METRIC_COST_TOTAL,
            description="Total cost in currency units",
            unit="{currency}",
        )

        self._cost_model = self._meter.create_counter(
            name=METRIC_COST_MODEL,
            description="Cost per model in currency units",
            unit="{currency}",
        )

        self._cost_tool = self._meter.create_counter(
            name=METRIC_COST_TOOL,
            description="Cost per tool in currency units",
            unit="{currency}",
        )

        # Token counters
        self._tokens_input = self._meter.create_counter(
            name=METRIC_TOKENS_INPUT,
            description="Total input tokens",
            unit="{token}",
        )

        self._tokens_output = self._meter.create_counter(
            name=METRIC_TOKENS_OUTPUT,
            description="Total output tokens",
            unit="{token}",
        )

        # Agent activity counters
        self._iterations = self._meter.create_counter(
            name=METRIC_AGENT_ITERATIONS,
            description="Total agent loop iterations",
            unit="{iteration}",
        )

        self._tool_calls = self._meter.create_counter(
            name=METRIC_AGENT_TOOL_CALLS,
            description="Total tool calls",
            unit="{call}",
        )

        self._runs = self._meter.create_counter(
            name=METRIC_AGENT_RUNS,
            description="Total agent runs",
            unit="{run}",
        )

        # Event counters
        self._downgrades = self._meter.create_counter(
            name=METRIC_COST_DOWNGRADE_EVENTS,
            description="Model downgrade events",
            unit="{event}",
        )

        self._rejections = self._meter.create_counter(
            name=METRIC_COST_REJECTION_EVENTS,
            description="Run rejection events",
            unit="{event}",
        )

        self._halts = self._meter.create_counter(
            name=METRIC_COST_HALT_EVENTS,
            description="Run halt events",
            unit="{event}",
        )

    def _get_base_attributes(self, context: RunContext) -> dict[str, str]:
        """Get base attributes for a metric from run context."""
        attrs = context.to_attributes()
        # Remove run_id if high cardinality is not desired
        if not self.config.include_run_id_attribute:
            attrs.pop("strands.run_id", None)
        return attrs

    def record_run_start(self, context: RunContext) -> None:
        """Record a run start event."""
        if not self._runs:
            return

        attrs = self._get_base_attributes(context)
        attrs["strands.event"] = "start"
        self._runs.add(1, attributes=attrs)

    def record_run_end(self, run_state: RunState) -> None:
        """Record a run end event with total costs."""
        attrs = self._get_base_attributes(run_state.context)
        attrs["strands.event"] = "end"
        attrs["strands.status"] = run_state.status

        if self._runs:
            self._runs.add(1, attributes=attrs)

        if self._cost_total and run_state.total_cost > 0:
            self._cost_total.add(run_state.total_cost, attributes=attrs)

        if self._tokens_input and run_state.total_input_tokens > 0:
            self._tokens_input.add(run_state.total_input_tokens, attributes=attrs)

        if self._tokens_output and run_state.total_output_tokens > 0:
            self._tokens_output.add(run_state.total_output_tokens, attributes=attrs)

    def record_model_cost(self, context: RunContext, usage: ModelUsage) -> None:
        """Record cost and tokens from a model call."""
        attrs = self._get_base_attributes(context)
        attrs["genai.model.name"] = usage.model_name

        if self._cost_model and usage.cost > 0:
            self._cost_model.add(usage.cost, attributes=attrs)

        if self._tokens_input and usage.prompt_tokens > 0:
            self._tokens_input.add(usage.prompt_tokens, attributes=attrs)

        if self._tokens_output and usage.completion_tokens > 0:
            self._tokens_output.add(usage.completion_tokens, attributes=attrs)

    def record_tool_cost(self, context: RunContext, usage: ToolUsage) -> None:
        """Record cost from a tool call."""
        attrs = self._get_base_attributes(context)
        attrs["strands.tool.name"] = usage.tool_name

        if self._tool_calls:
            self._tool_calls.add(1, attributes=attrs)

        if self._cost_tool and usage.cost > 0:
            self._cost_tool.add(usage.cost, attributes=attrs)

    def record_iteration(self, context: RunContext, usage: IterationUsage) -> None:
        """Record an iteration completion."""
        if not self._iterations:
            return

        attrs = self._get_base_attributes(context)
        attrs["strands.iteration_idx"] = str(usage.iteration_idx)
        self._iterations.add(1, attributes=attrs)

    def record_downgrade(
        self,
        context: RunContext,
        original_model: str,
        fallback_model: str,
        reason: str,
    ) -> None:
        """Record a model downgrade event."""
        if not self._downgrades:
            return

        attrs = self._get_base_attributes(context)
        attrs["genai.model.original"] = original_model
        attrs["genai.model.fallback"] = fallback_model
        attrs["strands.reason"] = reason[:100]  # Truncate for attribute limits
        self._downgrades.add(1, attributes=attrs)

    def record_rejection(self, context: RunContext, reason: str) -> None:
        """Record a run rejection event."""
        if not self._rejections:
            return

        attrs = self._get_base_attributes(context)
        attrs["strands.reason"] = reason[:100]
        self._rejections.add(1, attributes=attrs)

    def record_iteration_halt(self, context: RunContext, reason: str) -> None:
        """Record an iteration halt event."""
        if not self._halts:
            return

        attrs = self._get_base_attributes(context)
        attrs["strands.reason"] = reason[:100]
        self._halts.add(1, attributes=attrs)

    def shutdown(self) -> None:
        """Shutdown the meter provider and flush metrics."""
        if self._meter_provider:
            try:
                self._meter_provider.shutdown()
                logger.info("OpenTelemetry metrics shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down OpenTelemetry metrics: {e}")
