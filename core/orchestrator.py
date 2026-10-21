
import time
from pathlib import Path

from core.schemas import ExperimentPlan


class Orchestrator:
    def __init__(
        self,
        agent,
        runner,
        tracker,
        max_experiments=10,
        time_budget_sec=10800,
        final_reserve_sec=1800,
        history_path="artifacts/history.jsonl",
    ):
        if max_experiments < 1:
            raise ValueError("max_experiments must be positive")

        if time_budget_sec <= final_reserve_sec:
            raise ValueError("Time budget must exceed final reserve")

        self.agent = agent
        self.runner = runner
        self.tracker = tracker

        self.max_experiments = max_experiments
        self.time_budget_sec = time_budget_sec
        self.final_reserve_sec = final_reserve_sec
        self.history_path = Path(history_path)



    async def run(self, metric_name: str,time_budget_sec: float | None = None):
        budget = time_budget_sec if time_budget_sec is not None else self.time_budget_sec
        start = time.perf_counter()

        for iteration in range(self.max_experiments):
            elapsed = time.perf_counter() - start

            remaining = budget - (time.perf_counter() - start)
            research_time = remaining - self.final_reserve_sec

            if research_time <= 0:
                print("Research budget exhausted")
                break

            history = self.tracker.get_summary_for_agent()

            try:
                raw_plan = await self.agent.propose(history)
                plan = ExperimentPlan.model_validate(raw_plan)

            except Exception as e:
                print(
                    f"[iteration {iteration + 1}] "
                    f"Agent error: {type(e).__name__}: {e}"
                )
                break

            elapsed = time.perf_counter() - start
            research_time = (
                self.time_budget_sec
                - elapsed
                - self.final_reserve_sec
            )

            if plan.estimated_time_sec > research_time:
                print("Not enough time for another experiment")
                break

            result = self.runner.run(plan)

            self.tracker.add_experiment(plan, result)
            self.tracker.save_history(str(self.history_path))

            score = (
                result.metric.value
                if result.metric is not None
                else None
            )

            print(
                f"[iteration {iteration + 1}] "
                f"model={plan.model.name}, "
                f"status={result.status}, "
                f"score={score}"
            )

        return {
            "best_result": self.tracker.get_best(metric_name),
            "best_plan": self.tracker.get_best_plan(metric_name),
            "remaining_time_sec": max(
                0.0,
                self.time_budget_sec
                - (time.perf_counter() - start),
            ),
        }
