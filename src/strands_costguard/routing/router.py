"""Adaptive model router integrated with Cost Guard."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from strands_costguard.core.usage import ModelUsage

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelClient(Protocol):
    """Protocol for model clients that can be wrapped by the router."""

    def call(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make a model call and return the response."""
        ...


@dataclass
class RouterConfig:
    """Configuration for the model router."""

    retry_on_rate_limit: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    track_latency: bool = True


@dataclass
class ModelRouter:
    """
    Adaptive model router that integrates with Cost Guard.

    Wraps model calls to:
    - Apply Cost Guard decisions (model selection, token limits)
    - Record usage metrics after calls
    - Handle retries and fallbacks

    Example:
        ```python
        from strands_costguard import CostGuard, ModelRouter

        guard = CostGuard(config=...)
        router = ModelRouter(cost_guard=guard)

        # In your agent loop:
        decision = router.before_call(
            run_id=run_id,
            stage="planning",
            messages=messages,
        )

        if decision.allowed:
            response = your_model_client.call(
                model=decision.effective_model,
                messages=messages,
                max_tokens=decision.max_tokens,
            )
            router.after_call(run_id=run_id, response=response)
        ```
    """

    cost_guard: Any  # CostGuard - use Any to avoid circular import
    config: RouterConfig = field(default_factory=RouterConfig)
    _pending_calls: dict[str, dict] = field(default_factory=dict)

    def before_call(
        self,
        run_id: str,
        stage: str,
        messages: list[dict[str, Any]],
        requested_model: str | None = None,
    ) -> "ModelCallContext":
        """
        Prepare for a model call by consulting Cost Guard.

        Args:
            run_id: The run identifier
            stage: Semantic stage ("planning", "tool_selection", "synthesis", "other")
            messages: Messages to send to the model
            requested_model: Optionally specify the model (otherwise uses routing policy)

        Returns:
            ModelCallContext with effective model and settings.
        """
        # Estimate prompt tokens (rough approximation)
        prompt_tokens_estimate = self._estimate_tokens(messages)

        # Get model decision from Cost Guard
        model = requested_model or "gpt-4o-mini"  # Default if not specified
        decision = self.cost_guard.before_model_call(
            run_id=run_id,
            model_name=model,
            stage=stage,
            prompt_tokens_estimate=prompt_tokens_estimate,
        )

        context = ModelCallContext(
            run_id=run_id,
            stage=stage,
            requested_model=model,
            effective_model=decision.effective_model or model,
            max_tokens=decision.max_tokens,
            allowed=decision.allowed,
            was_downgraded=decision.was_downgraded,
            reason=decision.reason,
            warnings=decision.warnings,
            prompt_tokens_estimate=prompt_tokens_estimate,
        )

        if decision.allowed:
            self._pending_calls[run_id] = {
                "context": context,
                "start_time": time.time() if self.config.track_latency else None,
            }

        return context

    def after_call(
        self,
        run_id: str,
        response: dict[str, Any],
        model_name: str | None = None,
    ) -> ModelUsage:
        """
        Record usage after a model call completes.

        Args:
            run_id: The run identifier
            response: Response from the model call
            model_name: Model name (if different from decision)

        Returns:
            ModelUsage with recorded metrics.
        """
        pending = self._pending_calls.pop(run_id, None)
        start_time = pending.get("start_time") if pending else None
        context = pending.get("context") if pending else None

        # Extract usage from response
        usage_data = response.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        cached_tokens = usage_data.get("cached_tokens", 0)
        reasoning_tokens = usage_data.get("reasoning_tokens", 0)

        # Calculate latency
        latency_ms = None
        if start_time is not None:
            latency_ms = (time.time() - start_time) * 1000

        # Determine model name
        effective_model = model_name
        if not effective_model and context:
            effective_model = context.effective_model
        if not effective_model:
            effective_model = response.get("model", "unknown")

        usage = ModelUsage.from_response(
            model_name=effective_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        # Record with Cost Guard
        self.cost_guard.after_model_call(run_id=run_id, usage=usage)

        return usage

    def call(
        self,
        run_id: str,
        stage: str,
        messages: list[dict[str, Any]],
        client: ModelClient,
        requested_model: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], ModelUsage]:
        """
        Make a model call through the router.

        This is a convenience method that combines before_call, the actual call,
        and after_call into a single operation.

        Args:
            run_id: The run identifier
            stage: Semantic stage
            messages: Messages to send
            client: Model client to use for the call
            requested_model: Optionally specify the model
            **kwargs: Additional arguments to pass to the model client

        Returns:
            Tuple of (response, usage).

        Raises:
            RuntimeError: If the call is not allowed by Cost Guard.
        """
        context = self.before_call(
            run_id=run_id,
            stage=stage,
            messages=messages,
            requested_model=requested_model,
        )

        if not context.allowed:
            raise RuntimeError(f"Model call not allowed: {context.reason}")

        # Make the actual call
        response = client.call(
            messages=messages,
            model=context.effective_model,
            max_tokens=context.max_tokens,
            **kwargs,
        )

        usage = self.after_call(
            run_id=run_id,
            response=response,
            model_name=context.effective_model,
        )

        return response, usage

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Estimate the number of tokens in messages.

        This is a rough approximation using ~4 characters per token.
        For more accurate estimates, use tiktoken or similar.
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multi-part content
                for part in content:
                    if isinstance(part, dict):
                        total_chars += len(str(part.get("text", "")))

            # Add overhead for role and structure
            total_chars += 10

        # Rough estimate: ~4 characters per token
        return total_chars // 4


@dataclass
class ModelCallContext:
    """Context for a model call, returned by before_call."""

    run_id: str
    stage: str
    requested_model: str
    effective_model: str
    max_tokens: int | None
    allowed: bool
    was_downgraded: bool
    reason: str | None
    warnings: list[str]
    prompt_tokens_estimate: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "requested_model": self.requested_model,
            "effective_model": self.effective_model,
            "max_tokens": self.max_tokens,
            "allowed": self.allowed,
            "was_downgraded": self.was_downgraded,
            "reason": self.reason,
            "warnings": self.warnings,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
        }
