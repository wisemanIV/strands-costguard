"""Tests for policy storage and retrieval."""

from strands_costguard.policies.budget import BudgetScope
from strands_costguard.policies.store import (
    EnvPolicySource,
    FilePolicySource,
    PolicyStore,
)


class MockPolicySource:
    """Mock policy source for testing."""

    def __init__(
        self,
        budgets: list[dict] = None,
        routing_policies: list[dict] = None,
        pricing: dict = None,
    ):
        self._budgets = budgets or []
        self._routing = routing_policies or []
        self._pricing = pricing or {}

    def load_budgets(self) -> list[dict]:
        return self._budgets

    def load_routing_policies(self) -> list[dict]:
        return self._routing

    def load_pricing(self) -> dict:
        return self._pricing


class TestFilePolicySource:
    """Tests for FilePolicySource."""

    def test_load_yaml_file_not_found(self, tmp_path):
        """Should return empty dict when file not found."""
        source = FilePolicySource(path=str(tmp_path))
        assert source.load_budgets() == []
        assert source.load_routing_policies() == []
        assert source.load_pricing() == {}

    def test_load_budgets_from_yaml(self, tmp_path):
        """Should load budgets from YAML file."""
        budgets_content = """
budgets:
  - id: test-budget
    scope: tenant
    match:
      tenant_id: "*"
    max_cost: 100.0
"""
        (tmp_path / "budgets.yaml").write_text(budgets_content)
        source = FilePolicySource(path=str(tmp_path))

        budgets = source.load_budgets()
        assert len(budgets) == 1
        assert budgets[0]["id"] == "test-budget"
        assert budgets[0]["max_cost"] == 100.0

    def test_load_routing_from_yaml(self, tmp_path):
        """Should load routing policies from YAML file."""
        routing_content = """
routing_policies:
  - id: test-routing
    match:
      strand_id: "*"
    default_model: gpt-4o-mini
"""
        (tmp_path / "routing.yaml").write_text(routing_content)
        source = FilePolicySource(path=str(tmp_path))

        routing = source.load_routing_policies()
        assert len(routing) == 1
        assert routing[0]["id"] == "test-routing"
        assert routing[0]["default_model"] == "gpt-4o-mini"

    def test_load_pricing_from_yaml(self, tmp_path):
        """Should load pricing from YAML file."""
        pricing_content = """
pricing:
  models:
    custom-model:
      input_per_1k: 1.5
      output_per_1k: 3.0
"""
        (tmp_path / "pricing.yaml").write_text(pricing_content)
        source = FilePolicySource(path=str(tmp_path))

        pricing = source.load_pricing()
        assert "models" in pricing
        assert "custom-model" in pricing["models"]

    def test_load_empty_yaml(self, tmp_path):
        """Should handle empty YAML files."""
        (tmp_path / "budgets.yaml").write_text("")
        source = FilePolicySource(path=str(tmp_path))

        budgets = source.load_budgets()
        assert budgets == []

    def test_custom_filenames(self, tmp_path):
        """Should use custom filenames."""
        custom_budgets = """
budgets:
  - id: custom
    scope: tenant
    match:
      tenant_id: "*"
"""
        (tmp_path / "custom_budgets.yaml").write_text(custom_budgets)
        source = FilePolicySource(
            path=str(tmp_path),
            budgets_file="custom_budgets.yaml",
        )

        budgets = source.load_budgets()
        assert len(budgets) == 1
        assert budgets[0]["id"] == "custom"


class TestEnvPolicySource:
    """Tests for EnvPolicySource."""

    def test_load_budgets_from_env(self, monkeypatch):
        """Should load budget from environment variables."""
        monkeypatch.setenv("COST_GUARD_MAX_COST", "500.0")
        monkeypatch.setenv("COST_GUARD_PERIOD", "weekly")

        source = EnvPolicySource()
        budgets = source.load_budgets()

        assert len(budgets) == 1
        assert budgets[0]["id"] == "env-default"
        assert budgets[0]["max_cost"] == 500.0
        assert budgets[0]["period"] == "weekly"

    def test_load_budgets_no_env(self, monkeypatch):
        """Should return empty list when no env vars set."""
        # Clear the env var if it exists
        monkeypatch.delenv("COST_GUARD_MAX_COST", raising=False)

        source = EnvPolicySource()
        budgets = source.load_budgets()

        assert budgets == []

    def test_load_routing_from_env(self, monkeypatch):
        """Should load routing from environment variables."""
        monkeypatch.setenv("COST_GUARD_DEFAULT_MODEL", "gpt-4o")
        monkeypatch.setenv("COST_GUARD_FALLBACK_MODEL", "gpt-4o-mini")

        source = EnvPolicySource()
        routing = source.load_routing_policies()

        assert len(routing) == 1
        assert routing[0]["default_model"] == "gpt-4o"
        assert routing[0]["default_fallback_model"] == "gpt-4o-mini"

    def test_load_routing_no_env(self, monkeypatch):
        """Should return empty list when no env vars set."""
        monkeypatch.delenv("COST_GUARD_DEFAULT_MODEL", raising=False)

        source = EnvPolicySource()
        routing = source.load_routing_policies()

        assert routing == []

    def test_load_pricing_returns_empty(self):
        """Pricing via env vars not supported."""
        source = EnvPolicySource()
        assert source.load_pricing() == {}

    def test_custom_prefix(self, monkeypatch):
        """Should support custom environment variable prefix."""
        monkeypatch.setenv("MY_APP_MAX_COST", "200.0")

        source = EnvPolicySource(prefix="MY_APP_")
        budgets = source.load_budgets()

        assert len(budgets) == 1
        assert budgets[0]["max_cost"] == 200.0


class TestPolicyStore:
    """Tests for PolicyStore."""

    def test_initialize_loads_policies(self):
        """Should load policies on initialization."""
        budgets = [
            {
                "id": "test",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 100.0,
            }
        ]
        source = MockPolicySource(budgets=budgets)
        store = PolicyStore(source=source)

        assert len(store.budgets) == 1
        assert store.budgets[0].id == "test"

    def test_get_budgets_for_context(self):
        """Should return matching budgets for context."""
        budgets = [
            {
                "id": "tenant-budget",
                "scope": "tenant",
                "match": {"tenant_id": "prod"},
                "max_cost": 100.0,
            },
            {
                "id": "global-budget",
                "scope": "global",
                "match": {"tenant_id": "*"},
                "max_cost": 1000.0,
            },
        ]
        source = MockPolicySource(budgets=budgets)
        store = PolicyStore(source=source)

        # Should match both budgets for tenant "prod"
        matching = store.get_budgets_for_context("prod", "s1", "w1")
        assert len(matching) == 2

        # Should only match global budget for other tenants
        matching = store.get_budgets_for_context("dev", "s1", "w1")
        assert len(matching) == 1
        assert matching[0].id == "global-budget"

    def test_get_effective_budget(self):
        """Should return most specific budget."""
        budgets = [
            {
                "id": "workflow-budget",
                "scope": "workflow",
                "match": {"tenant_id": "prod", "strand_id": "agent1", "workflow_id": "flow1"},
                "max_cost": 10.0,
            },
            {
                "id": "tenant-budget",
                "scope": "tenant",
                "match": {"tenant_id": "prod"},
                "max_cost": 100.0,
            },
        ]
        source = MockPolicySource(budgets=budgets)
        store = PolicyStore(source=source)

        # Should get workflow-level budget (more specific)
        budget = store.get_effective_budget("prod", "agent1", "flow1")
        assert budget.id == "workflow-budget"

        # Should get tenant-level budget when workflow doesn't match
        budget = store.get_effective_budget("prod", "agent1", "other-flow")
        assert budget.id == "tenant-budget"

    def test_get_effective_budget_with_scope_filter(self):
        """Should filter by scope when requested."""
        budgets = [
            {
                "id": "tenant-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 100.0,
            },
            {
                "id": "strand-budget",
                "scope": "strand",
                "match": {"strand_id": "*"},
                "max_cost": 50.0,
            },
        ]
        source = MockPolicySource(budgets=budgets)
        store = PolicyStore(source=source)

        # Filter to tenant scope only
        budget = store.get_effective_budget("t1", "s1", "w1", scope=BudgetScope.TENANT)
        assert budget.id == "tenant-budget"

        # Filter to strand scope only
        budget = store.get_effective_budget("t1", "s1", "w1", scope=BudgetScope.STRAND)
        assert budget.id == "strand-budget"

    def test_get_routing_policy(self):
        """Should return matching routing policy."""
        routing = [
            {
                "id": "specific-routing",
                "match": {"strand_id": "agent1"},
                "default_model": "gpt-4o",
            },
            {
                "id": "default-routing",
                "match": {"strand_id": "*"},
                "default_model": "gpt-4o-mini",
            },
        ]
        source = MockPolicySource(routing_policies=routing)
        store = PolicyStore(source=source)

        # Should match specific routing
        policy = store.get_routing_policy("t1", "agent1", "w1")
        assert policy.id == "specific-routing"

        # Should match default routing
        policy = store.get_routing_policy("t1", "other-agent", "w1")
        assert policy.id == "default-routing"

    def test_get_routing_policy_no_match(self):
        """Should return None when no routing policy matches."""
        routing = [
            {
                "id": "specific-routing",
                "match": {"tenant_id": "prod"},
                "default_model": "gpt-4o",
            },
        ]
        source = MockPolicySource(routing_policies=routing)
        store = PolicyStore(source=source)

        policy = store.get_routing_policy("dev", "s1", "w1")
        assert policy is None

    def test_get_pricing(self):
        """Should return pricing data."""
        pricing = {"models": {"gpt-4o": {"input_per_1k": 2.5, "output_per_1k": 10.0}}}
        source = MockPolicySource(pricing=pricing)
        store = PolicyStore(source=source)

        assert store.get_pricing() == pricing

    def test_policy_priority_ordering(self):
        """Budgets should be ordered by priority (most specific first)."""
        budgets = [
            {
                "id": "global",
                "scope": "global",
                "match": {"tenant_id": "*"},
                "max_cost": 1000.0,
            },
            {
                "id": "workflow",
                "scope": "workflow",
                "match": {"tenant_id": "*", "strand_id": "*", "workflow_id": "*"},
                "max_cost": 10.0,
            },
            {
                "id": "tenant",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 100.0,
            },
        ]
        source = MockPolicySource(budgets=budgets)
        store = PolicyStore(source=source)

        # Workflow should come first, then strand/tenant, then global
        assert store.budgets[0].id == "workflow"
        assert store.budgets[1].id == "tenant"
        assert store.budgets[2].id == "global"

    def test_disabled_budget_not_matched(self):
        """Disabled budgets should not be matched."""
        budgets = [
            {
                "id": "disabled",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 100.0,
                "enabled": False,
            },
            {
                "id": "enabled",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "max_cost": 200.0,
                "enabled": True,
            },
        ]
        source = MockPolicySource(budgets=budgets)
        store = PolicyStore(source=source)

        matching = store.get_budgets_for_context("t1", "s1", "w1")
        assert len(matching) == 1
        assert matching[0].id == "enabled"
