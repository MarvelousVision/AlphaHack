import asyncio
import tempfile
from pathlib import Path

from core.experiment_runner import ExperimentRunner
from core.experiment_tracker import ExperimentTracker
from core.orchestrator import Orchestrator
from core.schemas import ExperimentPlan

import numpy as np
import pandas as pd

from solution import run_ml_pipeline


class MockAgent:
    async def propose(self, history):
        return {
            "action": "run_experiment",
            "hypothesis": "Test CatBoost model",
            "model": {
                "name": "cbr",
                "hyperparameters": {},
            },
            "preprocessing": {
                "missing_values": "native",
                "categorical_encoding": "native",
                "scaling": "none",
            },
            "features": {
                "strategy": "baseline",
                "drop_columns": [],
            },
            "reasoning": "Integration test",
            "estimated_time_sec": 10,
        }


def mock_ml_engine(plan):
    return {
        "score": 0.75,
        "metric_name": "rmse",
        "cv_std": 0.02,
    }
async def test_full_pipeline():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)

        sample_path = temp / "sample_submission.csv"
        output_path = temp / "submission.csv"

        pd.DataFrame({
            "id": [1, 2, 3],
            "target": [0.0, 0.0, 0.0],
        }).to_csv(sample_path, index=False)

        prepared_data = {"target_name": "target"}

        calls = []

        def mock_train_and_predict(plan, data):
            calls.append(plan.model.name)

            if len(calls) == 1:
                return np.array([0.1, 0.2, 0.3])

            return np.array([0.4, 0.5, 0.6])

        tracker = ExperimentTracker()
        runner = ExperimentRunner(executor=mock_ml_engine)

        orchestrator = Orchestrator(
            agent=MockAgent(),
            runner=runner,
            tracker=tracker,
            max_experiments=2,
            time_budget_sec=120,
            final_reserve_sec=20,
            history_path=str(temp / "history.jsonl"),
        )

        baseline_plan = ExperimentPlan.model_validate(
        await MockAgent().propose([])
    )

        result = await run_ml_pipeline(
            prepared_data=prepared_data,
            orchestrator=orchestrator,
            baseline_plan=baseline_plan,
            train_and_predict=mock_train_and_predict,
            sample_submission_path=str(sample_path),
            output_path=str(output_path),
            metric_name="rmse",
        )

        assert output_path.exists()

        submission = pd.read_csv(output_path)

        assert np.allclose(
            submission["target"].to_numpy(),
            [0.4, 0.5, 0.6],
        )

        assert len(calls) == 2
        assert result["best_plan"] is not None

        print("Full pipeline test PASSED!")

async def test_final_training_failure():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)

        sample_path = temp / "sample_submission.csv"
        output_path = temp / "submission.csv"

        pd.DataFrame({
            "id": [1, 2, 3],
            "target": [0.0, 0.0, 0.0],
        }).to_csv(sample_path, index=False)

        calls = 0

        def failing_train_and_predict(plan, data):
            nonlocal calls
            calls += 1

            if calls == 1:
                return np.array([0.1, 0.2, 0.3])

            raise RuntimeError("Simulated final training failure")

        tracker = ExperimentTracker()
        runner = ExperimentRunner(executor=mock_ml_engine)

        orchestrator = Orchestrator(
            agent=MockAgent(),
            runner=runner,
            tracker=tracker,
            max_experiments=1,
            time_budget_sec=120,
            final_reserve_sec=20,
            history_path=str(temp / "history.jsonl"),
        )

        baseline_plan = ExperimentPlan.model_validate(
            await MockAgent().propose([])
        )

        await run_ml_pipeline(
            prepared_data={"target_name": "target"},
            orchestrator=orchestrator,
            baseline_plan=baseline_plan,
            train_and_predict=failing_train_and_predict,
            sample_submission_path=str(sample_path),
            output_path=str(output_path),
            metric_name="rmse",
        )

        submission = pd.read_csv(output_path)

        assert calls == 2
        assert np.allclose(
            submission["target"].to_numpy(),
            [0.1, 0.2, 0.3],
        )

        print("Fallback test PASSED!")
async def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        tracker = ExperimentTracker()
        runner = ExperimentRunner(executor=mock_ml_engine)

        orchestrator = Orchestrator(
            agent=MockAgent(),
            runner=runner,
            tracker=tracker,
            max_experiments=2,
            time_budget_sec=120,
            final_reserve_sec=20,
            history_path=str(Path(temp_dir) / "history.jsonl"),
        )

        result = await orchestrator.run("rmse")

        assert result["best_plan"] is not None
        assert result["best_plan"].model.name == "cbr"
        assert len(tracker.get_history()) == 2
        assert (Path(temp_dir) / "history.jsonl").exists()

        print("Integration test PASSED!")
    await test_full_pipeline()
    await test_final_training_failure()


if __name__ == "__main__":
    asyncio.run(main())
    