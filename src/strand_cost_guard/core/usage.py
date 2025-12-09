"""Usage data structures for tracking resource consumption."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelUsage:
    """Usage metrics from a single model call."""

    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = 0.0
    latency_ms: Optional[float] = None
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_response(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float = 0.0,
        latency_ms: Optional[float] = None,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> "ModelUsage":
        """Create usage from model response data."""
        return cls(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )


@dataclass
class ToolUsage:
    """Usage metrics from a single tool call."""

    tool_name: str
    cost: float = 0.0
    latency_ms: Optional[float] = None
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    success: bool = True
    error_type: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class IterationUsage:
    """Aggregated usage metrics for a single agent loop iteration."""

    iteration_idx: int
    model_calls: list[ModelUsage] = field(default_factory=list)
    tool_calls: list[ToolUsage] = field(default_factory=list)
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_model_usage(self, usage: ModelUsage) -> None:
        """Add model usage to this iteration."""
        self.model_calls.append(usage)
        self.total_cost += usage.cost
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens

    def add_tool_usage(self, usage: ToolUsage) -> None:
        """Add tool usage to this iteration."""
        self.tool_calls.append(usage)
        self.total_cost += usage.cost

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this iteration."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def num_model_calls(self) -> int:
        """Number of model calls in this iteration."""
        return len(self.model_calls)

    @property
    def num_tool_calls(self) -> int:
        """Number of tool calls in this iteration."""
        return len(self.tool_calls)
