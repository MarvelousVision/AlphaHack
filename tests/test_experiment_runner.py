
from core.experiment_runner import ExperimentRunner
from core.schemas import (
    ExperimentPlan,
    ModelConfig,
    PreprocessingConfig,
    FeatureConfig,
)


def mock_unimodel(plan):
    print(f"Training model: {plan.model.name}")

    return {
        "score": 0.764,
        "metric_name": "f1_macro",
        "model_path": "artifacts/mock_model.joblib",
    }


plan = ExperimentPlan(
    action="run_experiment",
    hypothesis="CatBoost baseline",
    model=ModelConfig(
        name="catboost",
        hyperparameters={
            "iterations": 500,
            "depth": 6,
        },
    ),
    preprocessing=PreprocessingConfig(
        missing_values="native",
        categorical_encoding="native",
        scaling="none",
    ),
    features=FeatureConfig(
        strategy="baseline",
        drop_columns=[],
    ),
    reasoning="Test baseline performance",
    estimated_time_sec=300,
)

runner = ExperimentRunner(executor=mock_unimodel)

result = runner.run(plan)

def invalid_unimodel(plan):
    return {
        "score": float("nan"),
        "metric_name": "f1_macro",
        "model_path": "artifacts/model.joblib"
    }


runner = ExperimentRunner(executor=invalid_unimodel)
result = runner.run(plan)

print("Status:", result.status)
print("Error:", result.error)