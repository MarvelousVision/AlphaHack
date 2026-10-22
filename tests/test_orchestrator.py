
from core.schemas import ExperimentPlan
from core.experiment_runner import ExperimentRunner
from core.experiment_tracker import ExperimentTracker
from core.orchestrator import Orchestrator


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
        estimated_time_sec=30,
    )


class MockAgent:
    def __init__(self):
        self.models = ["catboost", "lightgbm", "xgboost"]
        self.index = 0

    def propose(self, history):
        model_name = self.models[self.index]
        self.index += 1
        return make_plan(model_name)


def mock_unimodel(plan):
    scores = {
        "catboost": 0.76,
        "lightgbm": 0.81,
        "xgboost": 0.78,
    }

    return {
        "score": scores[plan.model.name],
        "metric_name": "f1_macro",
    }


agent = MockAgent()
runner = ExperimentRunner(executor=mock_unimodel)
tracker = ExperimentTracker()

orchestrator = Orchestrator(
    agent=agent,
    runner=runner,
    tracker=tracker,
    max_experiments=3,
    time_budget_sec=300,
    history_path="artifacts/test_orchestrator_history.jsonl",
)

output = orchestrator.run("f1_macro")

assert output["best_result"] is not None
assert output["best_plan"] is not None
assert output["best_result"].metric.value == 0.81
assert output["best_plan"].model.name == "lightgbm"
assert len(tracker.get_history()) == 3

print("Best model:", output["best_plan"].model.name)
print("Best score:", output["best_result"].metric.value)
print("Experiments:", len(tracker.get_history()))
print("Orchestrator test PASSED!")
