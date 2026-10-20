# 1. Imports
import os
import tempfile
from pathlib import Path

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

    # Write a complete temporary file first.
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

        # Replace old submission only after successful writing.
        os.replace(tmp_path, output)

    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return output


# 3. Управление ML pipeline
def run_ml_pipeline(
    prepared_data,
    orchestrator,
    baseline_plan,
    train_and_predict,
    sample_submission_path,
    output_path,
    metric_name,
):
    # 1. Create baseline submission
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

    # 2. Run AutoResearch
    print("Starting AutoResearch...")
    research_output = orchestrator.run(metric_name)

    best_plan = research_output["best_plan"]

    # 3. If AutoResearch found a successful experiment
    if best_plan is not None:
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


# 4. Entry point организаторов
# Сюда позже подключим настоящий запуск
# согласно baseline.py
if __name__ == "__main__":
    print("Pipeline is ready. Waiting for the organizer's baseline.py integration.")
    
    # Пример того, как это будет выглядеть (псевдокод):
    # 1. prepared_data = load_data(...)
    # 2. executor = YarikMLEngine(...)
    # 3. runner = ExperimentRunner(executor)
    # 4. tracker = ExperimentTracker()
    # 5. agent = QwenAgent(...)
    # 6. orchestrator = Orchestrator(agent, runner, tracker)
    # 
    # run_ml_pipeline(
    #     prepared_data=prepared_data,
    #     orchestrator=orchestrator,
    #     baseline_plan=...,
    #     train_and_predict=...,
    #     sample_submission_path="data/sample_submission.csv",
    #     output_path="submissions/final.csv",
    #     metric_name="f1_macro",
    # )