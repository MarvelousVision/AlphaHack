
from core.experiment_tracker import ExperimentTracker
from core.schemas import ExperimentResult, MetricResult


def make_result(model_name, score, metric_name, direction):
    return ExperimentResult(
        experiment_id=model_name,
        status="success",
        model_name=model_name,
        metric=MetricResult(
            name=metric_name,
            value=score,
            direction=direction,
        ),
        duration_sec=1.0,
    )


tracker = ExperimentTracker()

# Здесь для проверки Tracker используем один тестовый план.
# Создай объект ExperimentPlan по нашей схеме.
plan = ...  

tracker.add_experiment(
    plan,
    make_result("catboost", 0.76, "f1_macro", "maximize")
)

tracker.add_experiment(
    plan,
    make_result("lightgbm", 0.81, "f1_macro", "maximize")
)

tracker.add_experiment(
    plan,
    make_result("xgboost", 0.78, "f1_macro", "maximize")
)

best = tracker.get_best("f1_macro")
tracker_rmse = ExperimentTracker()

for model, score in [
    ("catboost", 42.5),
    ("lightgbm", 39.2),
    ("xgboost", 41.7),
]:
    tracker_rmse.add_experiment(
        plan,
        make_result(model, score, "rmse", "minimize")
    )

best = tracker_rmse.get_best("rmse")

print("Best model:", best.model_name)
print("Best RMSE:", best.metric.value)

