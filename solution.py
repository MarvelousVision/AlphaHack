import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from agent.llm_agent import AutoResearchAgent
from core.schemas import ExperimentPlan
from core.experiment_runner import ExperimentRunner
from core.experiment_tracker import ExperimentTracker
from core.orchestrator import Orchestrator
from data_tools.preprocessing import prepare_data


TOTAL_BUDGET = 10800
FINAL_RESERVE = 1800


def save_submission(predictions, sample_submission_path, output_path, target_name):
    sample = pd.read_csv(sample_submission_path)
    values = np.asarray(predictions)

    if target_name not in sample or values.ndim != 1 or len(values) != len(sample):
        raise ValueError("Invalid submission shape or target column")
    if pd.isna(values).any():
        raise ValueError("Predictions contain missing values")
    if pd.api.types.is_numeric_dtype(values) and not np.isfinite(values).all():
        raise ValueError("Predictions contain infinite values")

    sample[target_name] = values
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", dir=output.parent,
            encoding="utf-8", newline="", delete=False,
        ) as tmp:
            tmp_name = tmp.name
            sample.to_csv(tmp, index=False)
        os.replace(tmp_name, output)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
    return output


async def run_ml_pipeline(
    runner, tracker, orchestrator, baseline_plan, train_and_predict,
    sample_path, output_path, target_name, metric_name, started_at,
):

    baseline_result = runner.run(baseline_plan)
    if baseline_result.status != "success":
        raise RuntimeError(f"Baseline failed: {baseline_result.error}")
    
    tracker.add_experiment(baseline_plan, baseline_result)
    tracker.save_history("artifacts/history.jsonl")

    predictions = train_and_predict(baseline_plan)
    save_submission(predictions, sample_path, output_path, target_name)
    print(f"Baseline saved: {output_path}")

    remaining = TOTAL_BUDGET - (time.perf_counter() - started_at)
    if remaining > FINAL_RESERVE:
        try:
            await orchestrator.run(metric_name, time_budget_sec=remaining)
        except Exception as exc:
            print(f"AutoResearch failed: {exc}. Keeping baseline.")

    best_plan = tracker.get_best_plan(metric_name)
    if best_plan is not None and best_plan != baseline_plan:
        if TOTAL_BUDGET - (time.perf_counter() - started_at) > 60:
            try:
                predictions = train_and_predict(best_plan)
                save_submission(predictions, sample_path, output_path, target_name)
                print("Best model submission saved.")
            except Exception as exc:
                print(f"Final training failed: {exc}. Keeping baseline.")

    return tracker.get_best(metric_name)


async def run_solution(task_dir, output_path):
    started_at = time.perf_counter()
    task_dir = Path(task_dir)
    sample_path = task_dir / "sample_submission.csv"

    X_train, y, X_test, numeric, cat, task, ids, tgt, id_name = prepare_data(str(task_dir))
    metric_name = "rmse" if task == "regression" else "f1_macro"
    engine_metric = "RMSE" if task == "regression" else "f1_macro"

    engine_history = Path("artifacts/experiments.json")
    engine_history.parent.mkdir(parents=True, exist_ok=True)
    engine_history.write_text("{}", encoding="utf-8")

    sys.modules.setdefault("solution", sys.modules[__name__])
    globals().update({
        "X_train": X_train.to_pandas(), "y": y.to_pandas(),
        "X_test": X_test.to_pandas(), "ids": ids.to_pandas(),
        "numeric": numeric, "cat": cat, "task": task,
        "tgt": tgt, "id_name": id_name, "METRIC": engine_metric,
        "WAY_TO_JSON": str(engine_history),
    })
    from data_tools.models import unimodel, the_end

    experiment_numbers = {}

    def key(plan):
        return plan.model_dump_json(include={"model", "features"})

    def execute_experiment(plan):
        before = set(json.loads(engine_history.read_text(encoding="utf-8")))

        params = plan.model.hyperparameters.copy()

        if plan.model.name in ("cbr", "cbc"):
            params["verbose"] = False

        history = unimodel(
            model=plan.model.name,
            to_drop=plan.features.drop_columns or None,
            **params,
        )
        new_keys = [k for k in history if str(k) not in before]
        if len(new_keys) != 1:
            raise RuntimeError("Could not identify new ML experiment")
        record = history[new_keys[0]]
        experiment_numbers[key(plan)] = int(new_keys[0])
        actual_metric = str(record["metric_name"]).lower()
        if actual_metric != metric_name:
            raise ValueError(f"Unexpected metric: {actual_metric}")
        return {
            "score": record["mean_score"],
            "metric_name": actual_metric,
            "cv_std": record["score_std"],
        }

    def train_and_predict(plan):
        number = experiment_numbers[key(plan)]
        submission = the_end(number)
        return submission[tgt].to_numpy()

    baseline_plan = ExperimentPlan.model_validate({
        "action": "run_experiment",
        "hypothesis": "Initial CatBoost baseline",
        "model": {
            "name": "cbr" if task == "regression" else "cbc",
            "hyperparameters": {"iterations": 100, "verbose": False},
        },
        "features": {"strategy": "baseline", "drop_columns": []},
        "reasoning": "Get a valid submission before AutoResearch",
        "estimated_time_sec": 600,
    })

    runner = ExperimentRunner(executor=execute_experiment)
    tracker = ExperimentTracker()
    description_path = task_dir / "task.md"
    task_info = {
        "task_type": task,
        "metric_name": metric_name,
        "description": description_path.read_text(encoding="utf-8")
        if description_path.exists() else "",
    }

    async with AsyncOpenAI() as client:
        agent = AutoResearchAgent(client=client, model_name=os.environ["OPENAI_MODEL"])
        orchestrator = Orchestrator(
            agent=agent, runner=runner, tracker=tracker, task_info=task_info,
            max_experiments=300, time_budget_sec=TOTAL_BUDGET,
            final_reserve_sec=FINAL_RESERVE,
            history_path="artifacts/history.jsonl",
        )
        return await run_ml_pipeline(
            runner, tracker, orchestrator, baseline_plan, train_and_predict,
            sample_path, output_path, tgt, metric_name, started_at,
        )


if __name__ == "__main__":
    import sys
    import asyncio

    task_path = Path(sys.argv[1]).resolve()

    asyncio.run(
        run_solution(
            task_dir=str(task_path.parent),
            output_path=str(task_path.parent / "submission.csv"),
        )
    )
