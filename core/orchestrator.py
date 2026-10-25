
import time
from pathlib import Path

from core.schemas import ExperimentPlan, EnsemblePlan


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
        estimated_final_train_sec=0,
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
        self.estimated_final_train_sec = max(
            0.0, float(estimated_final_train_sec)
        )
        self.history_path = Path(history_path)

    def estimate_duration(self, plan: ExperimentPlan) -> float:
        max_duration = self.tracker.get_max_model_duration(
            plan.model.name
        )

        estimated = float(plan.estimated_time_sec)

        if max_duration is not None:
            estimated = max(estimated, max_duration * 1.5)

        return estimated

    def get_final_reserve(self) -> float:
        durations = [
            entry["result"].duration_sec
            for entry in self.tracker.get_history()
            if entry["result"].status == "success"
        ]

        max_duration = max(durations, default=0)

        return max(
            self.final_reserve_sec,
            max_duration * 1.5,
            self.estimated_final_train_sec * 1.5 + 300,
        )

    async def run(
        self,
        metric_name: str,
        time_budget_sec: float | None = None,
        on_new_best=None,
    ):
        budget = (
            time_budget_sec
            if time_budget_sec is not None
            else self.time_budget_sec
        )

        start = time.perf_counter()
        consecutive_agent_errors = 0
        consecutive_duplicates = 0

        for iteration in range(self.max_experiments):
            remaining = budget - (time.perf_counter() - start)
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

            except Exception as exc:
                consecutive_agent_errors += 1

                print(
                    f"Agent error "
                    f"({consecutive_agent_errors}/3): {exc}"
                )

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
                consecutive_duplicates += 1

                print(
                    f"[iteration {iteration + 1}] "
                    f"Duplicate experiment "
                    f"({consecutive_duplicates}/3)"
                )

                if consecutive_duplicates >= 3:
                    print("Too many duplicates. Stopping research.")
                    break

                continue

            consecutive_duplicates = 0

            remaining = budget - (time.perf_counter() - start)
            research_time = remaining - self.get_final_reserve()

            if research_time <= 0:
                print("Research budget exhausted after agent response")
                break

            estimated_duration = self.estimate_duration(plan)

            required_time = estimated_duration * 1.5 + 120

            if required_time > research_time:
                print(
                    f"Not enough time for {plan.model.name}: "
                    f"required {required_time:.0f}s, "
                    f"available {research_time:.0f}s"
                )
                continue
            previous_best = self.tracker.get_best(metric_name)
            previous_score = (
                previous_best.metric.value
                if previous_best is not None and previous_best.metric is not None
                else None
            )
            experiment_start = time.perf_counter()
            result = self.runner.run(plan)
            experiment_duration = time.perf_counter() - experiment_start
            print(
                f"[TIMING] model={plan.model.name}, "
                f"experiment={experiment_duration:.1f}s, "
                f"remaining={budget - (time.perf_counter() - start):.1f}s",
                flush=True,
            )

            self.tracker.add_experiment(plan, result)
            if (
                on_new_best is not None
                and result.status == "success"
                and result.metric is not None
            ):
                new_score = result.metric.value
                direction = result.metric.direction

                is_better = (
                    previous_score is None
                    or (
                        new_score < previous_score
                        if direction == "minimize"
                        else new_score > previous_score
                    )
                )

                if is_better:
                    remaining_now = budget - (time.perf_counter() - start)

                    try:
                        on_new_best(plan, result, remaining_now)
                    except Exception as exc:
                        print(f"[SAVE BEST FAILED] {exc}", flush=True)

            try:
                self.tracker.save_history(str(self.history_path))
            except OSError as exc:
                print(f"Failed to save experiment history: {exc}")

            score = (
                result.metric.value
                if result.metric is not None
                else None
            )

            print(
                f"[iteration {iteration + 1}] "
                f"model={plan.model.name}, "
                f"status={result.status}, "
                f"score={score}",
                flush=True,
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
            "best_result": best_result,
            "best_plan": best_plan,
            "remaining_time_sec": max(
                0.0,
                budget - (time.perf_counter() - start),
            ),
        }
    async def propose_ensemble(self, remaining_time_sec: float):
        raw_plan = await self.agent.propose_ensemble(
            task_info=self.task_info,
        )
        plan = EnsemblePlan.model_validate(raw_plan)
        plan.validate_task(self.task_info["task_type"])
        return plan
    async def run_ensemble(
        self,
        raw_plan,
        choir,
        task_type: str,
        time_budget_sec: float | None = None,
    ):
        plan = EnsemblePlan.model_validate(raw_plan)
        plan.validate_task(task_type)

        budget = (
            self.time_budget_sec
            if time_budget_sec is None
            else float(time_budget_sec)
        )

        usable_budget = budget - 300

        if usable_budget <= 0:
            raise TimeoutError("Not enough time for ensemble")

        started_at = time.perf_counter()

        predictions = choir(
            *plan.models,
            time_left=usable_budget,
        )

        if predictions is None:
            raise RuntimeError("choir returned no predictions")

        elapsed = time.perf_counter() - started_at
        return {
            "plan": plan,
            "predictions": predictions,
            "duration_sec": elapsed,
            "remaining_time_sec": max(0.0, budget - elapsed),
        }