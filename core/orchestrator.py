
import time
from pathlib import Path

from core.schemas import ExperimentPlan


class Orchestrator:
    def __init__(
        self,
        agent,
        runner,
        tracker,
        task_info,
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
        self.task_info = task_info

        self.max_experiments = max_experiments
        self.time_budget_sec = time_budget_sec
        self.final_reserve_sec = final_reserve_sec
        self.history_path = Path(history_path)
    def estimate_duration(self, plan: ExperimentPlan) -> float:
        max_duration = self.tracker.get_max_model_duration(plan.model.name)

        if max_duration is None:
            return float(plan.estimated_time_sec)

        return max(
            float(plan.estimated_time_sec),
            max_duration * 1.2,
        )
    def get_final_reserve(self) -> float:
        durations = [
            entry["result"].duration_sec
            for entry in self.tracker.get_history()
            if entry["result"].status == "success"
        ]

        max_duration = max(durations, default=0)

        return max(self.final_reserve_sec, max_duration * 1.2)
    
    async def run(self, metric_name: str,time_budget_sec: float | None = None):
        budget = time_budget_sec if time_budget_sec is not None else self.time_budget_sec
        start = time.perf_counter()
        consecutive_agent_errors = 0
        for iteration in range(self.max_experiments):
            elapsed = time.perf_counter() - start

            remaining = budget - elapsed
            research_time = remaining - self.get_final_reserve() 

            if research_time <= 0:
                print("Research budget exhausted")
                break

            history = self.tracker.get_summary_for_agent()
            best = self.tracker.get_best(metric_name)

            best_result = (
                best.model_dump(mode="json")
                if best is not None
                else None
            )
            try:
                raw_plan = await self.agent.propose(
                    history=history,
                    best_result=best_result,
                    task_info=self.task_info,
                    remaining_time_sec=research_time,
                )
                plan = ExperimentPlan.model_validate(raw_plan)
                consecutive_agent_errors = 0
            except Exception as e:
                consecutive_agent_errors += 1
                print(f"Agent error ({consecutive_agent_errors}/3): {e}")

                if consecutive_agent_errors >= 3:
                    print("Agent unavailable. Stopping research.")
                    break

                continue
            already_tested = any(
                entry["plan"].model == plan.model
                and entry["plan"].features == plan.features
                for entry in self.tracker.get_history()
            )
            if already_tested:
                print(f"[iteration {iteration + 1}] Duplicate experiment skipped")
                continue            
            elapsed = time.perf_counter() - start
            research_time = budget - elapsed - self.get_final_reserve()
            if research_time <= 0:
                print("Research budget exhausted after agent response")
                break
            estimated_duration = self.estimate_duration(plan)

            if estimated_duration > research_time:
                print(
                    f"Not enough time for {plan.model.name}: "
                    f"estimated {estimated_duration:.0f}s, "
                    f"available {research_time:.0f}s"
                )
                continue

            result = self.runner.run(plan)

            self.tracker.add_experiment(plan, result)
            try:
                self.tracker.save_history(str(self.history_path))
            except OSError as exc:
                print(f"Failed to save experiment history: {exc}")
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
        best_plan = self.tracker.get_best_plan(metric_name)
        best_result = self.tracker.get_best(metric_name)

        if best_plan is not None and best_result is not None:
            print(
                f"[BEST] model={best_plan.model.name}, "
                f"metric={metric_name}, "
                f"score={best_result.metric.value:.4f}",
                flush=True,
            )

        return {
            "best_result": self.tracker.get_best(metric_name),
            "best_plan": self.tracker.get_best_plan(metric_name),
            "remaining_time_sec": max(
                0.0,
                budget - (time.perf_counter() - start),
            ),
        }
