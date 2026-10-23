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
    if baseline_result.status == "success":
        print(
            f"[BASELINE] Модель: {baseline_plan.model.name}, "
            f"Время: {baseline_result.duration_sec:.1f} сек",
            flush=True,
        )
    else:
        print(f"[BASELINE] Ошибка: {baseline_result.error}", flush=True)
    if baseline_result.status != "success":
        raise RuntimeError(f"Baseline failed: {baseline_result.error}")
    
    tracker.add_experiment(baseline_plan, baseline_result)
    tracker.save_history("artifacts/history.jsonl")

    predictions = train_and_predict(baseline_plan)
    save_submission(predictions, sample_path, output_path, target_name)
    print(f"Baseline saved: {output_path}")
    saved_plan = baseline_plan

    remaining = TOTAL_BUDGET - (time.perf_counter() - started_at)
    if remaining > FINAL_RESERVE and os.getenv("BASELINE_ONLY") != "1":
        try:
            await orchestrator.run(metric_name, time_budget_sec=remaining)
        except Exception as exc:
            print(f"AutoResearch failed: {exc}. Keeping baseline.")

    best_plan = tracker.get_best_plan(metric_name)
    best_result = tracker.get_best(metric_name)

    if best_plan is not None and best_result is not None:
        print(
            f"[FINAL] Лучшая модель: {best_plan.model.name}, "
            f"CV {metric_name}: {best_result.metric.value:.4f}",
            flush=True,
        )

    if best_plan is not None and best_plan != baseline_plan:
        remaining_time = TOTAL_BUDGET - (time.perf_counter() - started_at)
        
        max_duration = tracker.get_max_model_duration(best_plan.model.name)
        estimated_final_time = (
            max_duration * 1.5
            if max_duration is not None
            else 900
        )
        SAFETY_MARGIN = 300
        if remaining_time > estimated_final_time + SAFETY_MARGIN:
            try:
                predictions = train_and_predict(best_plan)
                save_submission(predictions, sample_path, output_path, target_name)
                saved_plan = best_plan

                print("Best model submission saved.")
                
            except Exception as exc:
                print(f"Final training failed: {exc}. Keeping baseline.")
        else:
            print(
                f"[FINAL SKIPPED] Осталось {remaining_time:.0f} сек, "
                f"оценка обучения {estimated_final_time:.0f} сек. "
                f"Сохраняем baseline.",
                flush=True,
            )
    print(
            f"[SUBMISSION] Сохранена модель: {saved_plan.model.name}",
            flush=True,
        )
    return tracker.get_best(metric_name)

def resolve_data_dir(task_dir: Path) -> Path:
    import re

    task_dir = Path(task_dir).resolve()
    task_file = task_dir / "task.md"

    if not task_file.is_file():
        raise FileNotFoundError(f"Не найден task.md: {task_file}")

    task_text = task_file.read_text(encoding="utf-8")

    paths = re.findall(
        r"[A-Za-z0-9_./\\-]+[/\\]train\.csv",
        task_text,
        flags=re.IGNORECASE,
    )

    for relative_path in paths:
        relative_path = relative_path.replace("\\", "/").lstrip("/")
        candidate = (task_dir / Path(relative_path).parent).resolve()

        # Запрещаем выход за пределы каталога задачи
        if not candidate.is_relative_to(task_dir):
            continue

        required = (
            "train.csv",
            "test.csv",
            "sample_submission.csv",
        )

        if all((candidate / name).is_file() for name in required):
            return candidate
    candidates = [
        p.parent.resolve()
        for p in task_dir.rglob("train.csv")
        if (p.parent / "test.csv").is_file()
        and (p.parent / "sample_submission.csv").is_file()
    ]

    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Не удалось однозначно определить набор данных для {task_file}. "
        f"Пути из инструкции: {paths}. "
        f"Найденные каталоги: {candidates}"
    )

async def run_solution(task_dir, output_path):
    import re 
    started_at = time.perf_counter()
    task_dir = Path(task_dir).resolve()

    data_dir = resolve_data_dir(task_dir)
    print(f"[DATA] Каталог данных: {data_dir}", flush=True)

    sample_path = data_dir / "sample_submission.csv"

    X_train, y, X_test, numeric, cat, task, ids, tgt, id_name = prepare_data(
        str(data_dir),
        use_interactions=True,
    )

    metric_name = "rmse" if task == "regression" else "f1_macro"
    engine_metric = "RMSE" if task == "regression" else "f1_macro"

    # Подготавливаем историю экспериментов
    engine_history = Path("artifacts/experiments.json")
    engine_history.parent.mkdir(parents=True, exist_ok=True)
    engine_history.write_text("{}", encoding="utf-8")

    # Передаём данные в ML Engine
    sys.modules.setdefault("solution", sys.modules[__name__])

    globals().update({
        "X_train": X_train.to_pandas(),
        "y": y.to_pandas(),
        "X_test": X_test.to_pandas(),
        "ids": ids.to_pandas(),
        "numeric": numeric,
        "cat": cat,
        "task": task,
        "tgt": tgt,
        "id_name": id_name,
        "METRIC": engine_metric,
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
            "hyperparameters": {
                "iterations": 100,
                "depth": 8,
                "learning_rate": 0.05,
                "random_seed": 42,
                "verbose": False,
            },
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
        "n_rows": len(X_train),  
        "description": description_path.read_text(encoding="utf-8")
        if description_path.exists() else "",
        }

    async with AsyncOpenAI() as client:
        agent = AutoResearchAgent(client=client, model_name=os.environ["OPENAI_MODEL"])
        orchestrator = Orchestrator(
            agent=agent, runner=runner, tracker=tracker, task_info=task_info,
            max_experiments=300,
            time_budget_sec=TOTAL_BUDGET,
            final_reserve_sec=FINAL_RESERVE,
            history_path="artifacts/history.jsonl",
        )
        return await run_ml_pipeline(
            runner, tracker, orchestrator, baseline_plan, train_and_predict,
            sample_path, output_path, tgt, metric_name, started_at,
        )


if __name__ == "__main__":
    import asyncio

    task_path = Path(sys.argv[1]).resolve()

    asyncio.run(
        run_solution(
            task_dir=str(task_path.parent),
            output_path=str(task_path.parent / "submission.csv"),
        )
    )
