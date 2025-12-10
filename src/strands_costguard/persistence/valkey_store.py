"""Valkey/Redis budget state persistence.

Simple persistence layer using Valkey (Redis-compatible) for budget state.
Stores budget usage as JSON with atomic increment operations for concurrent safety.

Usage:
    import valkey
    from strands_costguard.persistence import ValkeyBudgetStore

    client = valkey.Valkey(host="localhost", port=6379)
    store = ValkeyBudgetStore(client)

    # Pass to CostGuard
    config = CostGuardConfig(
        policy_source=FilePolicySource(path="./policies"),
        budget_store=store,
    )
    guard = CostGuard(config=config)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import valkey

logger = logging.getLogger(__name__)

# Key prefix for all budget data
KEY_PREFIX = "strands_costguard:budget:"


@dataclass
class BudgetStateData:
    """Budget state stored in Valkey."""

    budget_id: str
    scope_key: str
    period_start: str  # ISO format
    period_end: str  # ISO format
    total_cost: float = 0.0
    total_runs: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0
    model_costs: dict[str, float] = field(default_factory=dict)
    tool_costs: dict[str, float] = field(default_factory=dict)
    concurrent_run_ids: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data: str) -> "BudgetStateData":
        return cls(**json.loads(data))


class ValkeyBudgetStore:
    """Valkey/Redis persistence for budget state.

    Provides atomic operations for updating budget counters across
    multiple application instances.

    Key structure:
        strands_costguard:budget:{scope_key} -> JSON BudgetStateData

    Where scope_key follows the pattern:
        - global:{budget_id}
        - tenant:{tenant_id}:{budget_id}
        - strand:{tenant_id}:{strand_id}:{budget_id}
        - workflow:{tenant_id}:{strand_id}:{workflow_id}:{budget_id}
    """

    def __init__(
        self,
        client: "valkey.Valkey",
        key_prefix: str = KEY_PREFIX,
    ):
        """Initialize the store.

        Args:
            client: Valkey/Redis client instance
            key_prefix: Prefix for all keys (default: "strands_costguard:budget:")
        """
        self._client = client
        self._key_prefix = key_prefix

    def _make_key(self, scope_key: str) -> str:
        """Create full Redis key from scope key."""
        return f"{self._key_prefix}{scope_key}"

    def get(self, scope_key: str) -> BudgetStateData | None:
        """Get budget state for a scope.

        Args:
            scope_key: Budget scope key (e.g., "tenant:prod-001:monthly-budget")

        Returns:
            BudgetStateData if found, None otherwise
        """
        key = self._make_key(scope_key)
        data = self._client.get(key)
        if data is None:
            return None
        return BudgetStateData.from_json(data.decode("utf-8"))

    def set(
        self,
        scope_key: str,
        state: BudgetStateData,
        expire_at: datetime | None = None,
    ) -> None:
        """Store budget state.

        Args:
            scope_key: Budget scope key
            state: Budget state to store
            expire_at: Optional expiration time (e.g., end of budget period)
        """
        key = self._make_key(scope_key)
        self._client.set(key, state.to_json())
        if expire_at:
            self._client.expireat(key, expire_at)

    def delete(self, scope_key: str) -> bool:
        """Delete budget state.

        Args:
            scope_key: Budget scope key

        Returns:
            True if deleted, False if not found
        """
        key = self._make_key(scope_key)
        return self._client.delete(key) > 0

    def get_or_create(
        self,
        scope_key: str,
        budget_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> BudgetStateData:
        """Get existing state or create new one for the period.

        If existing state has expired (period_end in the past), creates fresh state.

        Args:
            scope_key: Budget scope key
            budget_id: Budget identifier
            period_start: Start of budget period
            period_end: End of budget period

        Returns:
            BudgetStateData (existing or newly created)
        """
        existing = self.get(scope_key)
        now = datetime.utcnow()

        if existing:
            # Check if period has expired
            existing_end = datetime.fromisoformat(existing.period_end)
            if now < existing_end:
                return existing
            # Period expired, will create new
            logger.info(f"Budget period expired for {scope_key}, resetting")

        # Create new state
        state = BudgetStateData(
            budget_id=budget_id,
            scope_key=scope_key,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
        )
        self.set(scope_key, state, expire_at=period_end)
        return state

    def increment_cost(
        self,
        scope_key: str,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_name: str | None = None,
        tool_name: str | None = None,
    ) -> BudgetStateData | None:
        """Atomically increment cost counters.

        Uses Valkey WATCH/MULTI/EXEC for optimistic locking.

        Args:
            scope_key: Budget scope key
            cost: Cost to add
            input_tokens: Input tokens to add
            output_tokens: Output tokens to add
            model_name: Model name for attribution
            tool_name: Tool name for attribution

        Returns:
            Updated state, or None if scope_key not found
        """
        key = self._make_key(scope_key)

        # Retry loop for optimistic locking
        for _ in range(3):
            try:
                self._client.watch(key)
                data = self._client.get(key)
                if data is None:
                    self._client.unwatch()
                    return None

                state = BudgetStateData.from_json(data.decode("utf-8"))
                state.total_cost += cost
                state.total_input_tokens += input_tokens
                state.total_output_tokens += output_tokens

                if model_name:
                    state.model_costs[model_name] = state.model_costs.get(model_name, 0.0) + cost
                if tool_name:
                    state.tool_costs[tool_name] = state.tool_costs.get(tool_name, 0.0) + cost
                    state.total_tool_calls += 1

                pipe = self._client.pipeline(True)
                pipe.set(key, state.to_json())
                pipe.execute()
                return state

            except Exception as e:
                # WatchError or other - retry
                logger.debug(f"Retry increment_cost due to: {e}")
                continue

        logger.warning(f"Failed to increment cost for {scope_key} after retries")
        return None

    def increment_run_count(self, scope_key: str, run_id: str) -> int | None:
        """Atomically increment run count and track concurrent run.

        Args:
            scope_key: Budget scope key
            run_id: Run ID to track

        Returns:
            New total run count, or None if scope_key not found
        """
        key = self._make_key(scope_key)

        for _ in range(3):
            try:
                self._client.watch(key)
                data = self._client.get(key)
                if data is None:
                    self._client.unwatch()
                    return None

                state = BudgetStateData.from_json(data.decode("utf-8"))
                state.total_runs += 1
                if run_id not in state.concurrent_run_ids:
                    state.concurrent_run_ids.append(run_id)

                pipe = self._client.pipeline(True)
                pipe.set(key, state.to_json())
                pipe.execute()
                return state.total_runs

            except Exception:
                continue

        return None

    def remove_concurrent_run(self, scope_key: str, run_id: str) -> int | None:
        """Remove a run from concurrent tracking.

        Args:
            scope_key: Budget scope key
            run_id: Run ID to remove

        Returns:
            Remaining concurrent run count, or None if scope_key not found
        """
        key = self._make_key(scope_key)

        for _ in range(3):
            try:
                self._client.watch(key)
                data = self._client.get(key)
                if data is None:
                    self._client.unwatch()
                    return None

                state = BudgetStateData.from_json(data.decode("utf-8"))
                if run_id in state.concurrent_run_ids:
                    state.concurrent_run_ids.remove(run_id)

                pipe = self._client.pipeline(True)
                pipe.set(key, state.to_json())
                pipe.execute()
                return len(state.concurrent_run_ids)

            except Exception:
                continue

        return None

    def get_concurrent_run_count(self, scope_key: str) -> int:
        """Get current concurrent run count.

        Args:
            scope_key: Budget scope key

        Returns:
            Number of concurrent runs, 0 if not found
        """
        state = self.get(scope_key)
        if state is None:
            return 0
        return len(state.concurrent_run_ids)

    def list_budgets(self, pattern: str = "*") -> list[str]:
        """List all budget scope keys matching a pattern.

        Args:
            pattern: Glob pattern (e.g., "tenant:prod-*")

        Returns:
            List of scope keys (without prefix)
        """
        full_pattern = f"{self._key_prefix}{pattern}"
        keys = self._client.keys(full_pattern)
        prefix_len = len(self._key_prefix)
        return [k.decode("utf-8")[prefix_len:] for k in keys]
