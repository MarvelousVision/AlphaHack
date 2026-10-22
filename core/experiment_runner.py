
import math
import time
import uuid

from pydantic import ValidationError

from core.schemas import ExperimentPlan, ExperimentResult, MetricResult


def _direction_for(metric_name: str) -> str:
    directions = {
        "f1_macro": "maximize",
        "accuracy": "maximize",
        "roc_auc": "maximize",
        "rmse": "minimize",
        "mae": "minimize",
        "mse": "minimize",
    }

    if metric_name not in directions:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return directions[metric_name]


class ExperimentRunner:
    def __init__(self, executor):
        self.executor = executor

    def _validate_engine_output(self, raw) -> str | None:
        if not isinstance(raw, dict):
            return "ML Engine output must be a dictionary"

        required = ("score", "metric_name")

        for key in required:
            if key not in raw:
                return f"Missing required field: {key}"

        score = raw["score"]

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return "Score must be a number"

        if not math.isfinite(score):
            return "Score must be finite"

        metric_name = raw["metric_name"]

        if not isinstance(metric_name, str):
            return "metric_name must be a string"

        try:
            _direction_for(metric_name)
        except ValueError as e:
            return str(e)

        return None

    def run(self, plan: dict | ExperimentPlan) -> ExperimentResult:
        experiment_id = uuid.uuid4().hex

        if isinstance(plan, dict):
            try:
                plan = ExperimentPlan.model_validate(plan)
            except ValidationError as e:
                return ExperimentResult(
                    experiment_id=experiment_id,
                    status="failed",
                    model_name="unknown",
                    duration_sec=0.0,
                    error=f"Invalid plan: {e}",
                )

        elif not isinstance(plan, ExperimentPlan):
            return ExperimentResult(
                experiment_id=experiment_id,
                status="failed",
                model_name="unknown",
                duration_sec=0.0,
                error="Plan must be dict or ExperimentPlan",
            )

        start = time.perf_counter()

        try:
            raw = self.executor(plan)

        except Exception as e:
            return ExperimentResult(
                experiment_id=experiment_id,
                status="failed",
                model_name=plan.model.name,
                duration_sec=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}",
            )

        problem = self._validate_engine_output(raw)

        if problem is not None:
            return ExperimentResult(
                experiment_id=experiment_id,
                status="failed",
                model_name=plan.model.name,
                duration_sec=time.perf_counter() - start,
                error=problem,
            )

        metric_name = raw["metric_name"]

        return ExperimentResult(
            experiment_id=experiment_id,
            status="success",
            model_name=plan.model.name,
            metric=MetricResult(
                name=metric_name,
                value=float(raw["score"]),
                direction=_direction_for(metric_name),
            ),
            cv_std=raw.get("cv_std"),
            duration_sec=time.perf_counter() - start,
            error=None,
        )
