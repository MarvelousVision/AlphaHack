
from core.ml_adapter import MLAdapter
from core.experiment_runner import ExperimentRunner
from core.schemas import ExperimentPlan


def mock_unimodel(plan, prepared_data):
    assert prepared_data["task_type"] == "multiclass"

    return {
        "score": 0.81,
        "metric_name": "f1_macro",
        "cv_std": 0.018,
    }


prepared_data = {
    "task_type": "multiclass",
    "X_train": "mock_features",
    "y_train": "mock_target",
}

plan = ExperimentPlan(
    action="run_experiment",
    hypothesis="Test CatBoost",
    model={
        "name": "catboost",
        "hyperparameters": {"depth": 6},
    },
    preprocessing={
        "missing_values": "native",
        "categorical_encoding": "native",
        "scaling": "none",
    },
    features={"strategy": "baseline", "drop_columns": []},
    reasoning="Check baseline quality",
    estimated_time_sec=300,
)

adapter = MLAdapter(mock_unimodel, prepared_data)
runner = ExperimentRunner(executor=adapter)

result = runner.run(plan)

assert result.status == "success"
assert result.metric.value == 0.81

print("MLAdapter test PASSED!")
