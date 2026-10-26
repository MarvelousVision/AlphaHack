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

from core.schemas import ExperimentPlan, OptunaExperimentPlan
from core.experiment_runner import ExperimentRunner
from core.experiment_tracker import ExperimentTracker
from core.orchestrator import Orchestrator
from data_tools.preprocessing import prepare_data

TOTAL_BUDGET = 10800
FINAL_RESERVE = 1800

def train_baseline_predictions(
    task,
    X_train,
    y_train,
    X_test,
    cat,
    iterations=300,
):
    from catboost import CatBoostRegressor, CatBoostClassifier
    train = X_train.to_pandas().copy()
    test = X_test.to_pandas().copy()
    target = np.asarray(y_train).ravel()
    for column in cat:
        train[column] = train[column].fillna("__missing__").astype(str)
        test[column] = test[column].fillna("__missing__").astype(str)

    model_class = (
        CatBoostRegressor
        if task == "regression"
        else CatBoostClassifier
    )

    model = model_class(
        iterations=150,
        depth=5,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )
    print(
        f"[BASELINE DATA] rows={len(train)}, "
        f"features={train.shape[1]}, "
        f"cat_features={len(cat)}, "
        f"iterations={model.get_params().get('iterations')}",
        flush=True,
    )
    model.fit(
        train,
        target,
        cat_features=cat,
    )
    predictions = np.asarray(model.predict(test)).ravel()
    if len(predictions) != len(test):
        raise ValueError("Baseline prediction count mismatch")

    return predictions
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
    runner, tracker, orchestrator, train_and_predict,
    sample_path, output_path, target_name, metric_name,
    started_at, baseline_duration,
):
    orchestrator.estimated_final_train_sec = baseline_duration
    saved_model = "cbr" if metric_name == "rmse" else "cbc"
    saved_score = None

    def on_new_best(plan, result, remaining_sec):
        nonlocal saved_model, saved_score

        if result.metric is None:
            return

        new_score = result.metric.value
        direction = result.metric.direction
        if saved_score is not None:
            is_better = (
                new_score < saved_score
                if direction == "minimize"
                else new_score > saved_score
            )
            if not is_better:
                return

        estimated = max(
            orchestrator.estimated_final_train_sec * 2,
            300,
        )
        safety_margin = 300

        if remaining_sec < estimated + safety_margin:
            print(
                f"[SAVE BEST SKIPPED] {plan.model.name}: "
                f"remaining={remaining_sec:.0f}s",
                flush=True,
            )
            return
        predictions = train_and_predict(
            plan,
            best_params=result.best_params,
        )

        save_submission(
            predictions,
            sample_path,
            output_path,
            target_name,
        )
        saved_model = plan.model.name
        saved_score = new_score
        print(
            f"[SAVE BEST] model={saved_model}, "
            f"score={saved_score:.5f}",
            flush=True,
        )

    remaining = TOTAL_BUDGET - (
        time.perf_counter() - started_at
    )
    if remaining > FINAL_RESERVE:
        try:
            await orchestrator.run(
                metric_name,
                time_budget_sec=remaining,
                on_new_best=on_new_best,
            )
        except Exception as exc:
            print(
                f"[AUTORESEARCH FAILED] {exc}",
                flush=True,
            )
    best_result = tracker.get_best(metric_name)

    if best_result is not None and best_result.metric is not None:
        print(
            f"[BEST CV] {best_result.model_name}: "
            f"{best_result.metric.value:.5f}",
            flush=True,
        )

    print(
        f"[SUBMISSION] Saved model: {saved_model}",
        flush=True,
    )

    return {
        "best_cv_result": best_result,
        "saved_model": saved_model,
        "saved_score": saved_score,
    }
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

        if not candidate.is_relative_to(task_dir):
            continue

        required = ("train.csv", "test.csv", "sample_submission.csv")

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
        f"Не удалось определить датасет для {task_file}. "
        f"Найденные каталоги: {candidates}"
    )

async def run_solution(task_dir, output_path):
    started_at = time.perf_counter()
    task_dir = Path(task_dir).resolve()

    data_dir = resolve_data_dir(task_dir)
    print(f"[DATA] Каталог данных: {data_dir}", flush=True)

    sample_path = data_dir / "sample_submission.csv"
    X_train, y, X_test, numeric, cat, task, ids, tgt, id_name = (
        prepare_data(str(data_dir))
    )

    metric_name = "rmse" if task == "regression" else "f1_macro"
    train_df = X_train.to_pandas()
    test_df = X_test.to_pandas()
    target = y[tgt].to_numpy().ravel()

    print("[BASELINE] Training CatBoost", flush=True)
    baseline_start = time.perf_counter()
    predictions = train_baseline_predictions(
        task=task,
        X_train=X_train,
        y_train=target,
        X_test=X_test,
        cat=cat,
    )

    save_submission(
        predictions,
        sample_path,
        output_path,
        tgt,
    )

    baseline_duration = time.perf_counter() - baseline_start
    print(
        f"[BASELINE] Submission saved in "
        f"{baseline_duration:.1f}s",
        flush=True,
    )

    if os.getenv("BASELINE_ONLY") == "1":
        return {"status": "baseline_saved"}
    remaining = TOTAL_BUDGET - (
        time.perf_counter() - started_at
    )

    if remaining <= FINAL_RESERVE:
        print("[AUTORESEARCH] Not enough time", flush=True)
        return {"status": "baseline_saved"}
    try:
        from agent.llm_agent import AutoResearchAgent
        from data_tools.models import unimodel, the_end
        import inspect

        parameters = inspect.signature(the_end).parameters
        required_parameters = {
            "model",
            "best_params",
            "train",
            "target",
            "test",
        }

        if not required_parameters.issubset(parameters):
            raise RuntimeError(
                "New the_end() interface is not ready"
            )
    except Exception as exc:
        print(
            f"[AUTORESEARCH UNAVAILABLE] {exc}",
            flush=True,
        )
        print("[FALLBACK] Keeping baseline", flush=True)
        return {"status": "baseline_saved", "error": str(exc)}
    
    def get_drop_columns(plan):
        requested = plan.drop_columns

        if not isinstance(requested, list):
            raise ValueError("drop_columns must be a list")

        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate columns in drop_columns")

        unknown = sorted(set(requested) - set(train_df.columns))
        if unknown:
            raise ValueError(f"Unknown drop_columns: {unknown}")

        if len(requested) > 5:
            raise ValueError("Too many drop_columns (maximum 5)")

        if len(requested) >= len(train_df.columns):
            raise ValueError("Cannot drop all features")

        return requested

    def execute_experiment(plan, time_budget_sec=None):
        if not isinstance(plan, OptunaExperimentPlan):
            raise TypeError("Expected OptunaExperimentPlan")
        if time_budget_sec is None or time_budget_sec <= 0:
            raise ValueError("Invalid Optuna time budget")
        search_space = convert_search_space(
            plan.model.search_space
        )
        raw = unimodel(
            model=plan.model.name,
            search_space=search_space,
            time_budget_sec=time_budget_sec,
            task=task,
            cat=cat,
            numeric=numeric,
            train=train_df,
            target=target,
            to_drop=get_drop_columns(plan),
        )
        return {
            **raw,
            "metric_name": metric_name,
        }

    def train_and_predict(plan, best_params=None):
        if not isinstance(plan, OptunaExperimentPlan):
            raise TypeError("Expected OptunaExperimentPlan")
        if best_params is None:
            raise ValueError("Missing best_params")
        predictions = the_end(
            model=plan.model.name,
            best_params=best_params,
            task=task,
            cat=cat,
            numeric=numeric,
            train=train_df,
            target=target,
            test=test_df,
            to_drop=get_drop_columns(plan),
        )
        predictions = np.asarray(predictions)

        if predictions.ndim != 1:
            raise ValueError(
                f"Expected 1D predictions, got {predictions.shape}"
            )
        if len(predictions) != len(test_df):
            raise ValueError(
                f"Expected {len(test_df)} predictions, got {len(predictions)}"
            )
        return predictions

    runner = ExperimentRunner(executor=execute_experiment)
    tracker = ExperimentTracker()
    description_path = task_dir / "task.md"

    task_info = {
        "task_type": task,
        "metric_name": metric_name,
        "n_rows": len(train_df),
        "feature_names": train_df.columns.tolist(),
        "description": (
            description_path.read_text(encoding="utf-8")
            if description_path.exists()
            else ""
        ),
    }
    try:
        async with AsyncOpenAI() as client:
            agent = AutoResearchAgent(
                client=client,
                model_name=os.environ["OPENAI_MODEL"],
            )
            orchestrator = Orchestrator(
                agent=agent,
                runner=runner,
                tracker=tracker,
                task_info=task_info,
                max_experiments=300,
                time_budget_sec=TOTAL_BUDGET,
                final_reserve_sec=FINAL_RESERVE,
                estimated_final_train_sec=baseline_duration,
                history_path="artifacts/history.jsonl",
            )

            return await run_ml_pipeline(
                runner,
                tracker,
                orchestrator,
                train_and_predict,
                sample_path,
                output_path,
                tgt,
                metric_name,
                started_at,
                baseline_duration,
            )

    except Exception as exc:
        print(
            f"[AUTORESEARCH FAILED] "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        print(
            "[FALLBACK] Keeping saved submission",
            flush=True,
        )
        return {
            "status": "fallback",
            "error": str(exc),
        }
    
if __name__ == "__main__":
    import asyncio

    task_path = Path(sys.argv[1]).resolve()

    asyncio.run(
        run_solution(
            task_dir=str(task_path.parent),
            output_path=str(task_path.parent / "submission.csv"),
        )
    )
