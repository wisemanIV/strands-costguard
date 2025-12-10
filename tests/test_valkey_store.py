"""Tests for Valkey/Redis budget state persistence."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from strands_costguard.persistence.valkey_store import (
    KEY_PREFIX,
    BudgetStateData,
    ValkeyBudgetStore,
)


class TestBudgetStateData:
    """Tests for BudgetStateData dataclass."""

    def test_to_json(self):
        """Should serialize to JSON."""
        state = BudgetStateData(
            budget_id="test-budget",
            scope_key="tenant:prod:budget-1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            total_cost=50.0,
            total_runs=10,
        )

        json_str = state.to_json()
        parsed = json.loads(json_str)

        assert parsed["budget_id"] == "test-budget"
        assert parsed["total_cost"] == 50.0
        assert parsed["total_runs"] == 10

    def test_from_json(self):
        """Should deserialize from JSON."""
        data = {
            "budget_id": "test-budget",
            "scope_key": "tenant:prod:budget-1",
            "period_start": "2024-01-01T00:00:00",
            "period_end": "2024-02-01T00:00:00",
            "total_cost": 75.5,
            "total_runs": 15,
            "total_input_tokens": 10000,
            "total_output_tokens": 5000,
            "total_iterations": 100,
            "total_tool_calls": 50,
            "model_costs": {"gpt-4o": 50.0, "gpt-4o-mini": 25.5},
            "tool_costs": {"web_search": 0.5},
            "concurrent_run_ids": ["run-1", "run-2"],
        }

        state = BudgetStateData.from_json(json.dumps(data))

        assert state.budget_id == "test-budget"
        assert state.total_cost == 75.5
        assert state.model_costs["gpt-4o"] == 50.0
        assert len(state.concurrent_run_ids) == 2

    def test_roundtrip_serialization(self):
        """Should survive roundtrip serialization."""
        original = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            total_cost=123.45,
            model_costs={"gpt-4o": 100.0},
            concurrent_run_ids=["run-abc"],
        )

        restored = BudgetStateData.from_json(original.to_json())

        assert restored.budget_id == original.budget_id
        assert restored.total_cost == original.total_cost
        assert restored.model_costs == original.model_costs
        assert restored.concurrent_run_ids == original.concurrent_run_ids


class TestValkeyBudgetStore:
    """Tests for ValkeyBudgetStore."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Valkey client."""
        return MagicMock()

    @pytest.fixture
    def store(self, mock_client):
        """Create a ValkeyBudgetStore with mock client."""
        return ValkeyBudgetStore(client=mock_client)

    def test_make_key(self, store):
        """Should create properly prefixed keys."""
        key = store._make_key("tenant:prod:budget-1")
        assert key == f"{KEY_PREFIX}tenant:prod:budget-1"

    def test_custom_key_prefix(self, mock_client):
        """Should support custom key prefix."""
        store = ValkeyBudgetStore(client=mock_client, key_prefix="my_app:")
        key = store._make_key("test")
        assert key == "my_app:test"

    def test_get_returns_state(self, store, mock_client):
        """Should return BudgetStateData when key exists."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            total_cost=50.0,
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")

        result = store.get("tenant:t1:b1")

        assert result is not None
        assert result.budget_id == "test"
        assert result.total_cost == 50.0

    def test_get_returns_none_when_not_found(self, store, mock_client):
        """Should return None when key doesn't exist."""
        mock_client.get.return_value = None

        result = store.get("nonexistent")

        assert result is None

    def test_set_stores_data(self, store, mock_client):
        """Should store state data."""
        state = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
        )

        store.set("tenant:t1:b1", state)

        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert f"{KEY_PREFIX}tenant:t1:b1" in call_args[0]

    def test_set_with_expiration(self, store, mock_client):
        """Should set expiration when provided."""
        state = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
        )
        expire_at = datetime(2024, 2, 1)

        store.set("tenant:t1:b1", state, expire_at=expire_at)

        mock_client.expireat.assert_called_once()

    def test_delete_existing_key(self, store, mock_client):
        """Should return True when key is deleted."""
        mock_client.delete.return_value = 1

        result = store.delete("tenant:t1:b1")

        assert result is True
        mock_client.delete.assert_called_once()

    def test_delete_nonexistent_key(self, store, mock_client):
        """Should return False when key doesn't exist."""
        mock_client.delete.return_value = 0

        result = store.delete("nonexistent")

        assert result is False

    def test_get_or_create_returns_existing(self, store, mock_client):
        """Should return existing state when valid."""
        future_end = (datetime.utcnow() + timedelta(days=7)).isoformat()
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end=future_end,
            total_cost=50.0,
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")

        result = store.get_or_create(
            scope_key="tenant:t1:b1",
            budget_id="test",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 2, 1),
        )

        assert result.total_cost == 50.0
        mock_client.set.assert_not_called()  # Shouldn't create new

    def test_get_or_create_creates_new(self, store, mock_client):
        """Should create new state when key doesn't exist."""
        mock_client.get.return_value = None

        result = store.get_or_create(
            scope_key="tenant:t1:b1",
            budget_id="test",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 2, 1),
        )

        assert result.budget_id == "test"
        assert result.total_cost == 0.0
        mock_client.set.assert_called_once()

    def test_get_or_create_resets_expired(self, store, mock_client):
        """Should reset state when period has expired."""
        past_end = (datetime.utcnow() - timedelta(days=1)).isoformat()
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2023-12-01T00:00:00",
            period_end=past_end,
            total_cost=100.0,  # Old cost that should be reset
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")

        result = store.get_or_create(
            scope_key="tenant:t1:b1",
            budget_id="test",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 2, 1),
        )

        assert result.total_cost == 0.0  # Reset
        mock_client.set.assert_called_once()

    def test_increment_cost(self, store, mock_client):
        """Should increment cost atomically."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            total_cost=50.0,
            total_input_tokens=1000,
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        result = store.increment_cost(
            scope_key="tenant:t1:b1",
            cost=10.0,
            input_tokens=500,
            output_tokens=200,
            model_name="gpt-4o",
        )

        assert result is not None
        assert result.total_cost == 60.0
        assert result.total_input_tokens == 1500
        assert result.total_output_tokens == 200
        assert result.model_costs["gpt-4o"] == 10.0
        mock_pipe.execute.assert_called_once()

    def test_increment_cost_with_tool(self, store, mock_client):
        """Should track tool costs and calls."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        result = store.increment_cost(
            scope_key="tenant:t1:b1",
            cost=0.05,
            tool_name="web_search",
        )

        assert result is not None
        assert result.tool_costs["web_search"] == 0.05
        assert result.total_tool_calls == 1

    def test_increment_cost_returns_none_when_not_found(self, store, mock_client):
        """Should return None when scope key doesn't exist."""
        mock_client.get.return_value = None

        result = store.increment_cost("nonexistent", cost=10.0)

        assert result is None

    def test_increment_run_count(self, store, mock_client):
        """Should increment run count and track concurrent run."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            total_runs=5,
            concurrent_run_ids=["run-1", "run-2"],
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        result = store.increment_run_count("tenant:t1:b1", "run-3")

        assert result == 6  # New total runs
        mock_pipe.execute.assert_called_once()

    def test_increment_run_count_no_duplicate(self, store, mock_client):
        """Should not add duplicate run IDs."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            total_runs=5,
            concurrent_run_ids=["run-1"],
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        # Try to add existing run ID
        store.increment_run_count("tenant:t1:b1", "run-1")

        # The set call should still happen but concurrent_run_ids should not grow
        mock_pipe.execute.assert_called_once()

    def test_remove_concurrent_run(self, store, mock_client):
        """Should remove run from concurrent tracking."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            concurrent_run_ids=["run-1", "run-2", "run-3"],
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        result = store.remove_concurrent_run("tenant:t1:b1", "run-2")

        assert result == 2  # Remaining concurrent runs
        mock_pipe.execute.assert_called_once()

    def test_remove_concurrent_run_not_present(self, store, mock_client):
        """Should handle removing non-existent run ID."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            concurrent_run_ids=["run-1"],
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        result = store.remove_concurrent_run("tenant:t1:b1", "run-nonexistent")

        assert result == 1  # Still 1 concurrent run
        mock_pipe.execute.assert_called_once()

    def test_get_concurrent_run_count(self, store, mock_client):
        """Should return current concurrent run count."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
            concurrent_run_ids=["run-1", "run-2", "run-3"],
        )
        mock_client.get.return_value = state_data.to_json().encode("utf-8")

        result = store.get_concurrent_run_count("tenant:t1:b1")

        assert result == 3

    def test_get_concurrent_run_count_not_found(self, store, mock_client):
        """Should return 0 when scope key doesn't exist."""
        mock_client.get.return_value = None

        result = store.get_concurrent_run_count("nonexistent")

        assert result == 0

    def test_list_budgets(self, store, mock_client):
        """Should list budget scope keys matching pattern."""
        mock_client.keys.return_value = [
            f"{KEY_PREFIX}tenant:prod:budget-1".encode(),
            f"{KEY_PREFIX}tenant:prod:budget-2".encode(),
        ]

        result = store.list_budgets("tenant:prod:*")

        assert len(result) == 2
        assert "tenant:prod:budget-1" in result
        assert "tenant:prod:budget-2" in result

    def test_list_budgets_empty(self, store, mock_client):
        """Should return empty list when no matches."""
        mock_client.keys.return_value = []

        result = store.list_budgets("nonexistent:*")

        assert result == []

    def test_retry_on_watch_error(self, store, mock_client):
        """Should retry on optimistic locking failures."""
        state_data = BudgetStateData(
            budget_id="test",
            scope_key="tenant:t1:b1",
            period_start="2024-01-01T00:00:00",
            period_end="2024-02-01T00:00:00",
        )

        # First two attempts fail, third succeeds
        mock_client.get.return_value = state_data.to_json().encode("utf-8")
        mock_pipe = MagicMock()

        # Simulate WatchError on first two attempts
        call_count = [0]

        def execute_side_effect():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("WatchError")
            return True

        mock_pipe.execute.side_effect = execute_side_effect
        mock_client.pipeline.return_value = mock_pipe

        result = store.increment_cost("tenant:t1:b1", cost=10.0)

        assert result is not None
        assert mock_pipe.execute.call_count == 3
