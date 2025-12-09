"""Policy storage and retrieval."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml

from strand_cost_guard.policies.budget import BudgetSpec, BudgetScope
from strand_cost_guard.policies.routing import RoutingPolicy

logger = logging.getLogger(__name__)


@dataclass
class FilePolicySource:
    """Load policies from YAML files."""

    path: str
    budgets_file: str = "budgets.yaml"
    routing_file: str = "routing.yaml"
    pricing_file: str = "pricing.yaml"

    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML file from the policy directory."""
        file_path = Path(self.path) / filename
        if not file_path.exists():
            logger.warning(f"Policy file not found: {file_path}")
            return {}
        with open(file_path, "r") as f:
            return yaml.safe_load(f) or {}

    def load_budgets(self) -> list[dict]:
        """Load budget specifications from YAML."""
        data = self._load_yaml(self.budgets_file)
        return data.get("budgets", [])

    def load_routing_policies(self) -> list[dict]:
        """Load routing policies from YAML."""
        data = self._load_yaml(self.routing_file)
        return data.get("routing_policies", [])

    def load_pricing(self) -> dict:
        """Load pricing table from YAML."""
        data = self._load_yaml(self.pricing_file)
        return data.get("pricing", {})


@dataclass
class EnvPolicySource:
    """Load policies from environment variables (for simple single-tenant setups)."""

    prefix: str = "COST_GUARD_"

    def load_budgets(self) -> list[dict]:
        """Load budget from environment variables."""
        import os

        max_cost = os.environ.get(f"{self.prefix}MAX_COST")
        if not max_cost:
            return []

        return [
            {
                "id": "env-default",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "period": os.environ.get(f"{self.prefix}PERIOD", "monthly"),
                "max_cost": float(max_cost),
            }
        ]

    def load_routing_policies(self) -> list[dict]:
        """Load routing from environment variables."""
        import os

        default_model = os.environ.get(f"{self.prefix}DEFAULT_MODEL")
        if not default_model:
            return []

        return [
            {
                "id": "env-default",
                "match": {"strand_id": "*"},
                "default_model": default_model,
                "default_fallback_model": os.environ.get(f"{self.prefix}FALLBACK_MODEL"),
            }
        ]

    def load_pricing(self) -> dict:
        """Pricing not supported via env vars - return empty."""
        return {}


@dataclass
class PolicyStore:
    """
    Central store for policies with caching and refresh capability.

    Policies are merged by priority:
    - workflow > strand > tenant > global defaults
    """

    source: FilePolicySource | EnvPolicySource
    refresh_interval_seconds: int = 300
    _budgets: list[BudgetSpec] = field(default_factory=list)
    _routing_policies: list[RoutingPolicy] = field(default_factory=list)
    _pricing: dict = field(default_factory=dict)
    _last_refresh: Optional[datetime] = None
    _snapshot: Optional[dict] = None  # Last known good config for fail-open

    def __post_init__(self) -> None:
        """Initialize by loading policies."""
        self.refresh()

    def refresh(self) -> None:
        """Reload policies from source."""
        try:
            budget_data = self.source.load_budgets()
            self._budgets = [BudgetSpec.from_dict(b) for b in budget_data]
            self._budgets.sort(key=lambda b: b.get_priority(), reverse=True)

            routing_data = self.source.load_routing_policies()
            self._routing_policies = [RoutingPolicy.from_dict(r) for r in routing_data]
            self._routing_policies.sort(key=lambda r: r.specificity_score(), reverse=True)

            self._pricing = self.source.load_pricing()

            self._last_refresh = datetime.utcnow()
            self._snapshot = {
                "budgets": budget_data,
                "routing": routing_data,
                "pricing": self._pricing,
            }
            logger.info(
                f"Loaded {len(self._budgets)} budgets, {len(self._routing_policies)} routing policies"
            )
        except Exception as e:
            logger.error(f"Failed to refresh policies: {e}")
            if self._snapshot:
                logger.warning("Using last known good configuration")
            else:
                raise

    def _maybe_refresh(self) -> None:
        """Refresh if interval has elapsed."""
        if self._last_refresh is None:
            self.refresh()
            return

        elapsed = datetime.utcnow() - self._last_refresh
        if elapsed > timedelta(seconds=self.refresh_interval_seconds):
            self.refresh()

    def get_budgets_for_context(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
    ) -> list[BudgetSpec]:
        """
        Get all applicable budgets for a context, ordered by priority.

        Higher priority budgets (more specific) come first.
        """
        self._maybe_refresh()
        return [
            budget
            for budget in self._budgets
            if budget.matches_context(tenant_id, strand_id, workflow_id)
        ]

    def get_effective_budget(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
        scope: Optional[BudgetScope] = None,
    ) -> Optional[BudgetSpec]:
        """
        Get the most specific (highest priority) budget for a context.

        Optionally filter by scope.
        """
        budgets = self.get_budgets_for_context(tenant_id, strand_id, workflow_id)
        if scope:
            budgets = [b for b in budgets if b.scope == scope]
        return budgets[0] if budgets else None

    def get_routing_policy(
        self,
        tenant_id: str,
        strand_id: str,
        workflow_id: str,
    ) -> Optional[RoutingPolicy]:
        """Get the most specific routing policy for a context."""
        self._maybe_refresh()
        for policy in self._routing_policies:
            if policy.matches_context(tenant_id, strand_id, workflow_id):
                return policy
        return None

    def get_pricing(self) -> dict:
        """Get the current pricing table."""
        self._maybe_refresh()
        return self._pricing

    @property
    def budgets(self) -> list[BudgetSpec]:
        """Get all loaded budgets."""
        return self._budgets

    @property
    def routing_policies(self) -> list[RoutingPolicy]:
        """Get all loaded routing policies."""
        return self._routing_policies
