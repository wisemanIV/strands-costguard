"""
Live agent example with real LLM API calls.

This example demonstrates CostGuard integration with a real Strands agent
making actual API calls to Anthropic's Claude.

Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - pip install strands-agents strands-costguard

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    python examples/live_agent_example.py
"""

import os
import sys
import uuid

from strands import Agent
from strands.models.anthropic import AnthropicModel

from strands_costguard import (
    CostGuard,
    CostGuardConfig,
    IterationUsage,
    ModelUsage,
)


class InlinePolicySource:
    """Inline policy source for the example."""

    def load_budgets(self) -> list[dict]:
        return [
            {
                "id": "example-budget",
                "scope": "tenant",
                "match": {"tenant_id": "*"},
                "period": "daily",
                "max_cost": 1.0,  # $1 daily limit for this example
                "soft_thresholds": [0.5, 0.8],
                "on_soft_threshold_exceeded": "LOG_ONLY",
                "hard_limit": True,
                "on_hard_limit_exceeded": "REJECT_NEW_RUNS",
                "constraints": {
                    "max_iterations_per_run": 5,
                    "max_tool_calls_per_run": 10,
                },
            }
        ]

    def load_routing_policies(self) -> list[dict]:
        return [
            {
                "id": "cost-aware-routing",
                "match": {"strand_id": "*"},
                "stages": [
                    {
                        "stage": "planning",
                        "default_model": "claude-haiku-4-5-20251001",
                        "max_tokens": 500,
                    },
                    {
                        "stage": "synthesis",
                        "default_model": "claude-sonnet-4-20250514",
                        "fallback_model": "claude-haiku-4-5-20251001",
                        "trigger_downgrade_on": {"soft_threshold_exceeded": True},
                    },
                ],
            }
        ]

    def load_pricing(self) -> dict:
        return {
            "currency": "USD",
            "fallback_input_per_1k": 1.0,
            "fallback_output_per_1k": 3.0,
            "models": {
                "claude-sonnet-4-20250514": {
                    "input_per_1k": 3.0,
                    "output_per_1k": 15.0,
                },
                "claude-haiku-4-5-20251001": {
                    "input_per_1k": 0.80,
                    "output_per_1k": 4.0,
                },
            },
        }


def run_cost_guarded_agent(task: str) -> dict:
    """
    Run a Strands agent with CostGuard enforcement.

    Args:
        task: The task for the agent to perform.

    Returns:
        Dictionary with result and cost information.
    """
    # Initialize CostGuard
    config = CostGuardConfig(
        policy_source=InlinePolicySource(),
        enable_budget_enforcement=True,
        enable_routing=True,
        enable_metrics=False,  # Disable OTEL for this example
    )
    guard = CostGuard(config=config)

    # Generate run identifiers
    run_id = str(uuid.uuid4())
    tenant_id = "example-tenant"
    strand_id = "live-agent"
    workflow_id = "demo-workflow"

    result = {
        "run_id": run_id,
        "status": "unknown",
        "output": None,
        "cost": 0.0,
        "model_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    # === Run Start ===
    print(f"\n{'=' * 60}")
    print(f"Starting run: {run_id[:8]}...")
    print(f"{'=' * 60}")

    admission = guard.on_run_start(
        tenant_id=tenant_id,
        strand_id=strand_id,
        workflow_id=workflow_id,
        run_id=run_id,
    )

    if not admission.allowed:
        print(f"Run rejected: {admission.reason}")
        result["status"] = "rejected"
        result["reason"] = admission.reason
        return result

    print(f"Run admitted. Budget remaining: ${admission.remaining_budget:.4f}")
    if admission.warnings:
        print(f"Warnings: {admission.warnings}")

    try:
        # === Iteration Loop ===
        for iteration in range(3):  # Max 3 iterations for demo
            # Check iteration limit
            iter_decision = guard.before_iteration(
                run_id=run_id,
                iteration_idx=iteration,
            )

            if not iter_decision.allowed:
                print(f"Iteration {iteration} halted: {iter_decision.reason}")
                result["status"] = "halted"
                break

            print(f"\n--- Iteration {iteration} ---")
            iteration_usage = IterationUsage(iteration_idx=iteration)

            # === Model Call ===
            model_decision = guard.before_model_call(
                run_id=run_id,
                model_name="claude-sonnet-4-20250514",
                stage="synthesis" if iteration > 0 else "planning",
                prompt_tokens_estimate=100,
            )

            if not model_decision.allowed:
                print(f"Model call blocked: {model_decision.reason}")
                continue

            effective_model = model_decision.effective_model
            print(f"Using model: {effective_model}")
            if model_decision.was_downgraded:
                print("  (Downgraded from requested model)")

            # Make real API call
            model = AnthropicModel(
                model_id=effective_model,
                max_tokens=model_decision.max_tokens or 1024,
            )
            agent = Agent(model=model)

            # Execute the task
            if iteration == 0:
                response = agent(task)
            else:
                response = agent("Continue or conclude the previous response briefly.")

            # Extract response text
            output_text = str(response) if response else ""
            print(f"Response: {output_text[:200]}{'...' if len(output_text) > 200 else ''}")

            # Extract actual token usage from the agent's metrics
            accumulated_usage = response.metrics.accumulated_usage
            prompt_tokens = accumulated_usage.get("inputTokens", 0)
            completion_tokens = accumulated_usage.get("outputTokens", 0)

            print(f"Tokens used: {prompt_tokens} input, {completion_tokens} output")

            usage = ModelUsage.from_response(
                model_name=effective_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            guard.after_model_call(run_id=run_id, usage=usage)
            iteration_usage.add_model_usage(usage)
            result["model_calls"] += 1
            result["total_input_tokens"] += prompt_tokens
            result["total_output_tokens"] += completion_tokens

            # Record iteration
            guard.after_iteration(
                run_id=run_id,
                iteration_idx=iteration,
                usage=iteration_usage,
            )

            # Store output from first iteration
            if iteration == 0:
                result["output"] = output_text

            # For demo, break after first meaningful response
            if len(output_text) > 50:
                result["status"] = "completed"
                break

    except Exception as e:
        print(f"Error during execution: {e}")
        result["status"] = "failed"
        result["error"] = str(e)

    finally:
        # Get final cost before ending run (on_run_end clears run state)
        result["cost"] = guard.get_run_cost(run_id) or 0.0

        # === Run End ===
        guard.on_run_end(
            run_id=run_id,
            status=result["status"],
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print("Run Summary")
        print(f"{'=' * 60}")
        print(f"Status: {result['status']}")
        print(f"Model calls: {result['model_calls']}")
        print(
            f"Total tokens: {result['total_input_tokens']} input, {result['total_output_tokens']} output"
        )
        print(f"Total cost: ${result['cost']:.6f}")

        # Budget summary
        summary = guard.get_budget_summary(tenant_id, strand_id, workflow_id)
        print("\nBudget Status:")
        for budget_id, stats in summary.items():
            print(f"  {budget_id}:")
            print(f"    Used: ${stats['current_cost']:.6f} / ${stats['max_cost']:.2f}")
            print(f"    Utilization: {stats['utilization']:.1%}")
            print(f"    Remaining: ${stats['remaining']:.6f}")

        guard.shutdown()

    return result


def main():
    """Main entry point."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo run this example:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print("  python examples/live_agent_example.py")
        sys.exit(1)

    print("Strands CostGuard - Live Agent Example")
    print("=" * 60)
    print("\nThis example demonstrates CostGuard with real API calls.")
    print("A small task will be sent to Claude to show cost tracking.\n")

    # Run the agent with a simple task
    task = "Explain what CostGuard is in exactly one sentence."

    result = run_cost_guarded_agent(task)

    print(f"\n{'=' * 60}")
    print("Example Complete!")
    print(f"{'=' * 60}")

    if result["output"]:
        print(f"\nAgent Output:\n{result['output']}")

    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
