from core.schemas import ExperimentPlan

from core.experiment_runner import ExperimentRunner
from core.experiment_tracker import ExperimentTracker



# 1. Mock функции Ярика
def mock_unimodel(plan):
    scores = {
        "catboost": 0.76,
        "lightgbm": 0.81,
        "xgboost": 0.78,
    }

    return {
        "score": scores[plan.model.name],
        "metric_name": "f1_macro",
        "model_path": f"artifacts/{plan.model.name}.joblib",
    }


# 2. Функция создания ExperimentPlan
def make_plan(model_name):
    return ExperimentPlan(
        action="run_experiment",
        hypothesis=f"Test {model_name}",
        model={
            "name": model_name,
            "hyperparameters": {},
        },
        preprocessing={
            "missing_values": "native",
            "categorical_encoding": "native",
            "scaling": "none",
        },
        features={
            "strategy": "baseline",
            "drop_columns": [],
        },
        reasoning="Compare model performance",
        estimated_time_sec=300,
    )


# 3. Создаём три плана
plan_catboost = make_plan("catboost")
plan_lightgbm = make_plan("lightgbm")
plan_xgboost = make_plan("xgboost")


# 4. Создаём Runner и Tracker
runner = ExperimentRunner(executor=mock_unimodel)
tracker = ExperimentTracker()


# 5. Запускаем три эксперимента
for plan in [plan_catboost, plan_lightgbm, plan_xgboost]:
    result = runner.run(plan)
    tracker.add_experiment(plan, result)

    print(
        f"Model: {result.model_name}, "
        f"Status: {result.status}, "
        f"Score: {result.metric.value if result.metric else None}"
    )


# 6. Выбираем лучший эксперимент
best = tracker.get_best("f1_macro")

print("\nBest model:", best.model_name)
print("Best score:", best.metric.value)
print("Total experiments:", len(tracker.get_history()))


# 7. Проверяем результат
assert best.model_name == "lightgbm"
assert best.metric.value == 0.81
assert len(tracker.get_history()) == 3

print("\nIntegration test PASSED!")
from pathlib import Path
import json

history_path = Path("artifacts/experiments.jsonl")

tracker.save_history(history_path)

assert history_path.exists()

with history_path.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

assert len(records) == 3
assert records[1]["plan"]["model"]["name"] == "lightgbm"
assert records[1]["result"]["metric"]["value"] == 0.81

print("JSONL test PASSED!")

summary = tracker.get_summary_for_agent()

assert len(summary) == 3
assert summary[0]["model"] == "catboost"
assert summary[1]["model"] == "lightgbm"
assert summary[1]["score"] == 0.81
assert summary[2]["model"] == "xgboost"

print("Agent summary test PASSED!")
print(summary)