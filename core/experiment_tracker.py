from core.schemas import (
    ExperimentPlan,
    OptunaExperimentPlan,
    ExperimentResult,
)
import json
from pathlib import Path

class ExperimentTracker:
    def __init__(self):
        self.history = []

    def add_experiment(
    self,
    plan: ExperimentPlan | OptunaExperimentPlan,
    result: ExperimentResult,
    ):
        self.history.append({"plan": plan, "result": result})
    def get_best(
    self,
    metric_name: str,
    ) -> ExperimentResult | None:
        candidates = [
            e["result"]
            for e in self.history
            if e["result"].status == "success"
            and e["result"].metric is not None
            and e["result"].metric.name == metric_name
        ]
        if not candidates:
            return None

        direction = candidates[0].metric.direction
        if direction == "maximize":
            return max(candidates, key=lambda r: r.metric.value)
        return min(candidates, key=lambda r: r.metric.value)
    
   
    def get_best_plan(
    self,
    metric_name: str,
    ) -> ExperimentPlan | OptunaExperimentPlan | None:
        best = self.get_best(metric_name)

        if best is None:
            return None

        for entry in self.history:
            if entry["result"].experiment_id == best.experiment_id:
                return entry["plan"]

        return None

   
    def get_history(self) -> list:
        return list(self.history)
    def save_history(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in self.history:
                record = {
                    "plan": entry["plan"].model_dump(mode="json"),
                    "result": entry["result"].model_dump(mode="json"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_summary_for_agent(self) -> list[dict]:
        summary = []
        for entry in self.history:
            plan = entry["plan"]
            result = entry["result"]
            metric = result.metric
            search_space = getattr(plan.model, "search_space", None)
            summary.append({
                "experiment_id": result.experiment_id,
                "model": plan.model.name,
                "hypothesis": plan.hypothesis,
                "status": result.status,
                "search_space": (
                    {
                        name: spec.model_dump(mode="json")
                        for name, spec in search_space.items()
                    }
                    if search_space is not None
                    else None
                ),
                "hyperparameters": getattr(
                    plan.model, "hyperparameters", None
                ),
                "metric": (
                    metric.model_dump(mode="json")
                    if metric is not None
                    else None
                ),
                "score": metric.value if metric is not None else None,
                "metric_name": metric.name if metric is not None else None,
                "best_params": result.best_params,
                "completed_trials": result.completed_trials,
                "top_trials": [
                    trial.model_dump(mode="json")
                    for trial in result.top_trials
                ],
                "drop_columns": getattr(plan, "drop_columns", []),
                "cv_std": result.cv_std,
                "duration_sec": result.duration_sec,
                "error": result.error,
            })

        return summary
    def get_model_durations(self) -> dict[str, float]:
        durations = {}

        for entry in self.history:
            model = entry["plan"].model.name
            result = entry["result"]

            if result.status != "success":
                continue

            durations.setdefault(model, []).append(result.duration_sec)

        return {
            model: sum(values) / len(values)
            for model, values in durations.items()
        }
    def get_max_model_duration(self, model_name: str) -> float | None:
        durations = [
            entry["result"].duration_sec
            for entry in self.history
            if entry["plan"].model.name == model_name
            and entry["result"].status == "success"
        ]

        return max(durations) if durations else None