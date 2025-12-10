"""
Example integration of Cost Guard with a Strands agent runtime.

This example shows how Cost Guard hooks would be called from within
a Strands agent loop, demonstrating the full lifecycle integration.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from strands.telemetry.config import StrandsTelemetry

from strands_costguard import (
    CostGuard,
    CostGuardConfig,
    FilePolicySource,
    IterationUsage,
    ModelRouter,
    ModelUsage,
    ToolUsage,
)


@dataclass
class AgentConfig:
    """Configuration for a Strands agent."""

    tenant_id: str
    strand_id: str
    workflow_id: str
    max_iterations: int = 10


class CostGuardedAgent:
    """
    Example Strands agent with Cost Guard integration.

    This shows how Cost Guard hooks integrate into the agent lifecycle.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        cost_guard: CostGuard,
        model_router: ModelRouter | None = None,
    ):
        self.config = agent_config
        self.cost_guard = cost_guard
        self.model_router = model_router or ModelRouter(cost_guard=cost_guard)
        self.run_id: str | None = None

    def run(self, task: str, metadata: dict | None = None) -> dict[str, Any]:
        """
        Execute an agent run with Cost Guard enforcement.

        Args:
            task: The task to execute
            metadata: Optional metadata for the run

        Returns:
            Result dictionary with output and cost information.
        """
        self.run_id = str(uuid.uuid4())

        # === Run Start ===
        admission = self.cost_guard.on_run_start(
            tenant_id=self.config.tenant_id,
            strand_id=self.config.strand_id,
            workflow_id=self.config.workflow_id,
            run_id=self.run_id,
            metadata=metadata,
        )

        if not admission.allowed:
            return {
                "status": "rejected",
                "reason": admission.reason,
                "cost": 0.0,
            }

        result = {"status": "running", "output": None, "iterations": 0}

        try:
            result = self._execute_loop(task)
        except Exception as e:
            result = {
                "status": "failed",
                "error": str(e),
                "iterations": result.get("iterations", 0),
            }
        finally:
            # === Run End ===
            self.cost_guard.on_run_end(
                run_id=self.run_id,
                status=result["status"],
            )
            result["cost"] = self.cost_guard.get_run_cost(self.run_id) or 0.0

        return result

    def _execute_loop(self, task: str) -> dict[str, Any]:
        """Execute the agent loop with Cost Guard checks."""
        messages = [{"role": "user", "content": task}]
        iteration_usage = IterationUsage(iteration_idx=0)

        for iteration in range(self.config.max_iterations):
            # === Before Iteration ===
            iter_decision = self.cost_guard.before_iteration(
                run_id=self.run_id,
                iteration_idx=iteration,
                context={"messages_count": len(messages)},
            )

            if not iter_decision.allowed:
                return {
                    "status": "halted",
                    "reason": iter_decision.reason,
                    "iterations": iteration,
                }

            iteration_usage = IterationUsage(iteration_idx=iteration)

            # === Planning Stage (Model Call) ===
            plan_usage = self._call_model(
                messages=messages,
                stage="planning",
            )
            if plan_usage:
                iteration_usage.add_model_usage(plan_usage)

            # Simulate getting tool calls from model response
            tool_calls = self._extract_tool_calls(plan_usage)

            if not tool_calls:
                # No tools needed, generate final response
                synthesis_usage = self._call_model(
                    messages=messages,
                    stage="synthesis",
                )
                if synthesis_usage:
                    iteration_usage.add_model_usage(synthesis_usage)

                # === After Iteration ===
                self.cost_guard.after_iteration(
                    run_id=self.run_id,
                    iteration_idx=iteration,
                    usage=iteration_usage,
                )

                return {
                    "status": "completed",
                    "output": "Task completed successfully",
                    "iterations": iteration + 1,
                }

            # === Tool Selection Stage ===
            for tool_call in tool_calls:
                tool_usage = self._call_tool(tool_call["name"], tool_call["args"])
                if tool_usage:
                    iteration_usage.add_tool_usage(tool_usage)

            # === After Iteration ===
            self.cost_guard.after_iteration(
                run_id=self.run_id,
                iteration_idx=iteration,
                usage=iteration_usage,
            )

        return {
            "status": "max_iterations",
            "output": "Reached maximum iterations",
            "iterations": self.config.max_iterations,
        }

    def _call_model(
        self,
        messages: list[dict],
        stage: str,
    ) -> ModelUsage | None:
        """Make a model call through Cost Guard."""
        # === Before Model Call ===
        decision = self.cost_guard.before_model_call(
            run_id=self.run_id,
            model_name="gpt-4o",  # Requested model
            stage=stage,
            prompt_tokens_estimate=self._estimate_tokens(messages),
        )

        if not decision.allowed:
            return None

        # Simulate model call with effective model
        effective_model = decision.effective_model

        # In real implementation, call the actual model here
        # response = model_client.call(model=effective_model, messages=messages)

        # Simulate response
        usage = ModelUsage.from_response(
            model_name=effective_model,
            prompt_tokens=self._estimate_tokens(messages),
            completion_tokens=150,
            latency_ms=500.0,
        )

        # === After Model Call ===
        self.cost_guard.after_model_call(
            run_id=self.run_id,
            usage=usage,
        )

        return usage

    def _call_tool(self, tool_name: str, args: dict) -> ToolUsage | None:
        """Make a tool call through Cost Guard."""
        # === Before Tool Call ===
        decision = self.cost_guard.before_tool_call(
            run_id=self.run_id,
            tool_name=tool_name,
        )

        if not decision.allowed:
            return None

        # In real implementation, call the actual tool here
        # result = tool_executor.call(tool_name, args)

        # Simulate tool execution
        usage = ToolUsage(
            tool_name=tool_name,
            latency_ms=100.0,
            input_size_bytes=len(str(args)),
            output_size_bytes=500,
            success=True,
        )

        # === After Tool Call ===
        self.cost_guard.after_tool_call(
            run_id=self.run_id,
            tool_name=tool_name,
            cost_metadata=usage,
        )

        return usage

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for messages."""
        total = sum(len(str(m.get("content", ""))) for m in messages)
        return total // 4  # Rough estimate

    def _extract_tool_calls(self, usage: ModelUsage | None) -> list[dict]:
        """Extract tool calls from model response (simulated)."""
        # In real implementation, parse from model response
        # For demo, return empty on first iteration
        return []


def main():
    """Run the example integration."""
    # Configure StrandsTelemetry first
    # For local testing without an OTEL collector, use console exporter
    telemetry = StrandsTelemetry()
    telemetry.setup_console_exporter()  # Prints spans to console
    telemetry.setup_meter(enable_console_exporter=True)  # Prints metrics to console

    # Initialize Cost Guard - uses global MeterProvider from StrandsTelemetry
    config = CostGuardConfig(
        policy_source=FilePolicySource(path="./policies"),
        enable_budget_enforcement=True,
        enable_routing=True,
        enable_metrics=True,
    )

    cost_guard = CostGuard(config=config)

    # Create agent
    agent_config = AgentConfig(
        tenant_id="prod-tenant-001",
        strand_id="analytics_assistant",
        workflow_id="data_analysis",
    )

    agent = CostGuardedAgent(
        agent_config=agent_config,
        cost_guard=cost_guard,
    )

    # Run agent
    print("Starting agent run...")
    result = agent.run(
        task="Analyze the sales data for Q4 2024",
        metadata={"user_id": "user-123"},
    )

    print(f"\nResult: {result}")

    # Get budget summary
    summary = cost_guard.get_budget_summary(
        agent_config.tenant_id,
        agent_config.strand_id,
        agent_config.workflow_id,
    )
    print("\nBudget Summary:")
    for budget_id, stats in summary.items():
        print(f"  {budget_id}: {stats['utilization']:.1%} utilized")

    # Cleanup - Cost Guard no longer manages OTEL lifecycle
    cost_guard.shutdown()


if __name__ == "__main__":
    main()
