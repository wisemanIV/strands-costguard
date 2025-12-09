"""Pricing table and cost computation."""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Default pricing for common models (USD per 1K tokens)
DEFAULT_MODEL_PRICING = {
    # OpenAI GPT-4 series
    "gpt-4": {"input_per_1k": 30.00, "output_per_1k": 60.00},
    "gpt-4-turbo": {"input_per_1k": 10.00, "output_per_1k": 30.00},
    "gpt-4-turbo-preview": {"input_per_1k": 10.00, "output_per_1k": 30.00},
    "gpt-4o": {"input_per_1k": 2.50, "output_per_1k": 10.00},
    "gpt-4o-mini": {"input_per_1k": 0.15, "output_per_1k": 0.60},
    "gpt-4.1": {"input_per_1k": 5.00, "output_per_1k": 15.00},
    "gpt-4.1-mini": {"input_per_1k": 0.40, "output_per_1k": 1.60},
    # OpenAI GPT-3.5 series
    "gpt-3.5-turbo": {"input_per_1k": 0.50, "output_per_1k": 1.50},
    "gpt-3.5-turbo-16k": {"input_per_1k": 3.00, "output_per_1k": 4.00},
    # Anthropic Claude series
    "claude-3-opus": {"input_per_1k": 15.00, "output_per_1k": 75.00},
    "claude-3-sonnet": {"input_per_1k": 3.00, "output_per_1k": 15.00},
    "claude-3-haiku": {"input_per_1k": 0.25, "output_per_1k": 1.25},
    "claude-3.5-sonnet": {"input_per_1k": 3.00, "output_per_1k": 15.00},
    "claude-3.5-haiku": {"input_per_1k": 0.80, "output_per_1k": 4.00},
    # Google Gemini series
    "gemini-1.5-pro": {"input_per_1k": 3.50, "output_per_1k": 10.50},
    "gemini-1.5-flash": {"input_per_1k": 0.075, "output_per_1k": 0.30},
    "gemini-2.0-flash": {"input_per_1k": 0.10, "output_per_1k": 0.40},
    # Meta Llama series (typical hosted pricing)
    "llama-3.1-405b": {"input_per_1k": 5.00, "output_per_1k": 15.00},
    "llama-3.1-70b": {"input_per_1k": 0.90, "output_per_1k": 0.90},
    "llama-3.1-8b": {"input_per_1k": 0.20, "output_per_1k": 0.20},
}


@dataclass
class ModelPricing:
    """Pricing configuration for a single model."""

    model_name: str
    input_per_1k: float
    output_per_1k: float
    currency: str = "USD"
    cached_input_per_1k: Optional[float] = None  # Discounted rate for cached tokens
    reasoning_per_1k: Optional[float] = None  # Rate for reasoning tokens (o1, etc.)

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> float:
        """Calculate the total cost for a model call."""
        # Standard input tokens (minus cached)
        standard_input_tokens = max(0, prompt_tokens - cached_tokens)
        input_cost = (standard_input_tokens / 1000) * self.input_per_1k

        # Cached tokens (if pricing specified)
        if cached_tokens > 0 and self.cached_input_per_1k is not None:
            input_cost += (cached_tokens / 1000) * self.cached_input_per_1k

        # Output tokens
        output_cost = (completion_tokens / 1000) * self.output_per_1k

        # Reasoning tokens (if applicable)
        reasoning_cost = 0.0
        if reasoning_tokens > 0 and self.reasoning_per_1k is not None:
            reasoning_cost = (reasoning_tokens / 1000) * self.reasoning_per_1k

        return input_cost + output_cost + reasoning_cost

    def estimate_cost(self, estimated_tokens: int, is_input: bool = True) -> float:
        """Estimate cost for a given token count."""
        rate = self.input_per_1k if is_input else self.output_per_1k
        return (estimated_tokens / 1000) * rate


@dataclass
class ToolPricing:
    """Pricing configuration for a tool."""

    tool_name: str
    cost_per_call: float = 0.0
    cost_per_input_byte: float = 0.0
    cost_per_output_byte: float = 0.0
    currency: str = "USD"

    def calculate_cost(
        self,
        input_size_bytes: int = 0,
        output_size_bytes: int = 0,
    ) -> float:
        """Calculate the cost for a tool call."""
        return (
            self.cost_per_call
            + (input_size_bytes * self.cost_per_input_byte)
            + (output_size_bytes * self.cost_per_output_byte)
        )


@dataclass
class PricingTable:
    """
    Central pricing table for models and tools.

    Supports loading from configuration and provides cost calculation.
    """

    currency: str = "USD"
    models: dict[str, ModelPricing] = field(default_factory=dict)
    tools: dict[str, ToolPricing] = field(default_factory=dict)
    fallback_input_per_1k: float = 1.0  # Default if model not found
    fallback_output_per_1k: float = 3.0

    def __post_init__(self) -> None:
        """Initialize with default pricing if empty."""
        if not self.models:
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default model pricing."""
        for model_name, pricing in DEFAULT_MODEL_PRICING.items():
            self.models[model_name] = ModelPricing(
                model_name=model_name,
                input_per_1k=pricing["input_per_1k"],
                output_per_1k=pricing["output_per_1k"],
                currency=self.currency,
            )

    def get_model_pricing(self, model_name: str) -> ModelPricing:
        """
        Get pricing for a model, with fallback for unknown models.

        Attempts fuzzy matching for model variants (e.g., gpt-4-0613 -> gpt-4).
        """
        # Exact match
        if model_name in self.models:
            return self.models[model_name]

        # Try prefix matching for versioned models
        for known_model in self.models:
            if model_name.startswith(known_model):
                return self.models[known_model]

        # Fallback pricing
        logger.warning(
            f"No pricing found for model '{model_name}', using fallback pricing"
        )
        return ModelPricing(
            model_name=model_name,
            input_per_1k=self.fallback_input_per_1k,
            output_per_1k=self.fallback_output_per_1k,
            currency=self.currency,
        )

    def get_tool_pricing(self, tool_name: str) -> ToolPricing:
        """Get pricing for a tool, returning zero-cost default if not configured."""
        if tool_name in self.tools:
            return self.tools[tool_name]
        return ToolPricing(tool_name=tool_name, currency=self.currency)

    def calculate_model_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> float:
        """Calculate cost for a model call."""
        pricing = self.get_model_pricing(model_name)
        return pricing.calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def calculate_tool_cost(
        self,
        tool_name: str,
        input_size_bytes: int = 0,
        output_size_bytes: int = 0,
    ) -> float:
        """Calculate cost for a tool call."""
        pricing = self.get_tool_pricing(tool_name)
        return pricing.calculate_cost(
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
        )

    def estimate_model_cost(
        self,
        model_name: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 0,
    ) -> float:
        """Estimate cost for a model call before execution."""
        pricing = self.get_model_pricing(model_name)
        input_cost = pricing.estimate_cost(estimated_input_tokens, is_input=True)
        output_cost = pricing.estimate_cost(estimated_output_tokens, is_input=False)
        return input_cost + output_cost

    @classmethod
    def from_dict(cls, data: dict) -> "PricingTable":
        """Create a PricingTable from a dictionary (e.g., from YAML)."""
        currency = data.get("currency", "USD")
        models = {}
        tools = {}

        for model_name, model_data in data.get("models", {}).items():
            models[model_name] = ModelPricing(
                model_name=model_name,
                input_per_1k=model_data["input_per_1k"],
                output_per_1k=model_data["output_per_1k"],
                currency=model_data.get("currency", currency),
                cached_input_per_1k=model_data.get("cached_input_per_1k"),
                reasoning_per_1k=model_data.get("reasoning_per_1k"),
            )

        for tool_name, tool_data in data.get("tools", {}).items():
            tools[tool_name] = ToolPricing(
                tool_name=tool_name,
                cost_per_call=tool_data.get("cost_per_call", 0.0),
                cost_per_input_byte=tool_data.get("cost_per_input_byte", 0.0),
                cost_per_output_byte=tool_data.get("cost_per_output_byte", 0.0),
                currency=tool_data.get("currency", currency),
            )

        table = cls(
            currency=currency,
            models=models,
            tools=tools,
            fallback_input_per_1k=data.get("fallback_input_per_1k", 1.0),
            fallback_output_per_1k=data.get("fallback_output_per_1k", 3.0),
        )

        # Merge with defaults for any missing models
        if not models:
            table._load_defaults()

        return table
