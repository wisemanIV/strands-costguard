"""
Basic usage example for Strand Cost Guard.

This example demonstrates how to integrate Cost Guard into a Strands-based
agent runtime with budget enforcement and cost tracking.
"""

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


def main():
    # Configure StrandsTelemetry first (handles all OTEL setup)
    telemetry = StrandsTelemetry()
    telemetry.setup_otlp_exporter(endpoint="http://localhost:4317")
    telemetry.setup_meter(enable_otlp_exporter=False)

    # Initialize Cost Guard with file-based policies
    # Cost Guard will use the global MeterProvider from StrandsTelemetry
    # Use path relative to this script's location
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    policies_path = os.path.join(script_dir, "policies")

    config = CostGuardConfig(
        policy_source=FilePolicySource(path=policies_path),
        enable_budget_enforcement=True,
        enable_routing=True,
        enable_metrics=True,
    )

    guard = CostGuard(config=config)

    # Optional: Create a model router for easier integration
    _router = ModelRouter(cost_guard=guard)

    # Simulate an agent run
    tenant_id = "prod-tenant-001"
    strand_id = "analytics_assistant"
    workflow_id = "data_analysis"
    run_id = "run-12345"

    # Start a new run
    admission = guard.on_run_start(
        tenant_id=tenant_id,
        strand_id=strand_id,
        workflow_id=workflow_id,
        run_id=run_id,
        metadata={"user_id": "user-abc", "request_type": "analysis"},
    )

    if not admission.allowed:
        print(f"Run rejected: {admission.reason}")
        return

    print(f"Run admitted. Remaining budget: ${admission.remaining_budget:.2f}")
    if admission.warnings:
        print(f"Warnings: {admission.warnings}")

    # Simulate agent loop iterations
    for iteration in range(3):
        # Check if iteration can proceed
        iter_decision = guard.before_iteration(
            run_id=run_id,
            iteration_idx=iteration,
        )

        if not iter_decision.allowed:
            print(f"Iteration halted: {iter_decision.reason}")
            break

        print(f"\n--- Iteration {iteration} ---")

        # Simulate a model call (planning stage)
        model_decision = guard.before_model_call(
            run_id=run_id,
            model_name="gpt-4o",
            stage="planning",
            prompt_tokens_estimate=500,
        )

        if model_decision.allowed:
            print(f"Model call allowed: {model_decision.effective_model}")
            if model_decision.was_downgraded:
                print(f"  (Downgraded: {model_decision.warnings})")

            # Simulate model response
            guard.after_model_call(
                run_id=run_id,
                usage=ModelUsage.from_response(
                    model_name=model_decision.effective_model,
                    prompt_tokens=500,
                    completion_tokens=200,
                ),
            )

        # Simulate a tool call
        tool_decision = guard.before_tool_call(
            run_id=run_id,
            tool_name="web_search",
        )

        if tool_decision.allowed:
            print(f"Tool call allowed. Remaining: {tool_decision.remaining_tool_calls}")

            # Simulate tool response
            guard.after_tool_call(
                run_id=run_id,
                tool_name="web_search",
                cost_metadata=ToolUsage(tool_name="web_search"),
            )

        # Record iteration completion
        guard.after_iteration(
            run_id=run_id,
            iteration_idx=iteration,
            usage=IterationUsage(iteration_idx=iteration),
        )

    # End the run
    guard.on_run_end(run_id=run_id, status="completed")

    # Get final cost
    print("\n--- Run Complete ---")
    print(f"Total cost: ${guard.get_run_cost(run_id) or 0:.4f}")

    # Get budget summary
    summary = guard.get_budget_summary(tenant_id, strand_id, workflow_id)
    print("\nBudget Summary:")
    for budget_id, stats in summary.items():
        print(f"  {budget_id}:")
        print(f"    Utilization: {stats['utilization']:.1%}")
        print(f"    Remaining: ${stats['remaining']:.2f}")

    # Cleanup - metrics are flushed via StrandsTelemetry
    guard.shutdown()


if __name__ == "__main__":
    main()
