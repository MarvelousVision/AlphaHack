# 1. Imports
import os
import tempfile
from pathlib import Path
import time 

import numpy as np
import pandas as pd

from core.experiment_runner import ExperimentRunner
from core.experiment_tracker import ExperimentTracker
from core.orchestrator import Orchestrator

def save_submission(
    predictions,
    sample_submission_path: str,
    output_path: str,
    target_name: str,
) -> Path:
    sample = pd.read_csv(sample_submission_path)

    if target_name not in sample.columns:
        raise ValueError(f"Unknown target column: {target_name}")
    values = np.asarray(predictions)

    if values.ndim != 1:
        raise ValueError("Predictions must be one-dimensional")

    if len(values) != len(sample):
        raise ValueError("Prediction count does not match test rows")

    if pd.isna(values).any():
        raise ValueError("Predictions contain missing values")

    if pd.api.types.is_numeric_dtype(values):
        if not np.isfinite(values).all():
            raise ValueError("Predictions contain infinite values")

    submission = sample.copy()
    submission[target_name] = values

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=".submission_",
            dir=output.parent,
            encoding="utf-8",
            newline="",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            submission.to_csv(tmp, index=False)
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return output

async def run_ml_pipeline(
    prepared_data,
    orchestrator,
    baseline_plan,
    train_and_predict,
    sample_submission_path,
    output_path,
    metric_name,
): 
    pipeline_start = time.perf_counter()
    print("Training baseline model...")
    baseline_predictions = train_and_predict(
        baseline_plan,
        prepared_data,
    )

    save_submission(
        predictions=baseline_predictions,
        sample_submission_path=sample_submission_path,
        output_path=output_path,
        target_name=prepared_data["target_name"],
    )
    print("Baseline submission saved.")
    TOTAL_BUDGET = 10800
    remaining_budget = max(
        0.0,
        TOTAL_BUDGET - (time.perf_counter() - pipeline_start)
    )

    print("Starting AutoResearch...")
    try:
        research_output = await orchestrator.run(
        metric_name,
        time_budget_sec=remaining_budget,
         )
    except Exception as e:
        print(f"AutoResearch failed: {type(e).__name__}: {e}")
        return {
            "best_plan": None,
            "best_result": None,
        }
        
    best_plan = research_output["best_plan"]
    remaining_total = TOTAL_BUDGET - (
    time.perf_counter() - pipeline_start
    )
    if best_plan is not None and remaining_total > 60:
        print(f"Best plan found: {best_plan.model.name}. Training on full data...")
        try:
            final_predictions = train_and_predict(
                best_plan,
                prepared_data,
            )

            save_submission(
                predictions=final_predictions,
                sample_submission_path=sample_submission_path,
                output_path=output_path,
                target_name=prepared_data["target_name"],
            )

            print("Final submission saved successfully.")

        except Exception as e:
            print(
                f"Final training failed: {e}. "
                "Keeping baseline submission."
            )
    else:
        print("AutoResearch did not find a better plan. Keeping baseline submission.")

    return research_output


if __name__ == "__main__":
    print("Pipeline is ready. Waiting for the organizer's baseline.py integration.")
