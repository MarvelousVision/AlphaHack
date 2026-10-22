
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from pydantic import BaseModel, Field, ConfigDict, model_validator

PARAM_RANGES = {
    "dtr": {"max_depth": (3, 30), "min_samples_leaf": (1, 20)},
    "dtc": {"max_depth": (3, 30), "min_samples_leaf": (1, 20)},

    "rfr": {
        "n_estimators": (100, 500),
        "max_depth": (5, 30),
        "min_samples_leaf": (1, 10),
    },
    "rfc": {
        "n_estimators": (100, 500),
        "max_depth": (5, 30),
        "min_samples_leaf": (1, 10),
    },

    "svr": {"C": (0.01, 100), "epsilon": (0.01, 1)},
    "svc": {"C": (0.01, 100)},

    "lvr": {"C": (0.01, 100), "epsilon": (0, 0.5)},
    "lvc": {"C": (0.01, 100)},

    "lbr": {
        "num_leaves": (15, 127),
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.2),
        "n_estimators": (200, 1500),
        "min_child_samples": (10, 100),
        "reg_lambda": (0, 10),
    },
    "lbc": {
        "num_leaves": (15, 127),
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.2),
        "n_estimators": (200, 1500),
        "min_child_samples": (10, 100),
        "reg_lambda": (0, 10),
    },

    "xbr": {
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.2),
        "n_estimators": (200, 1200),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_lambda": (0, 10),
    },
    "xbc": {
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.2),
        "n_estimators": (200, 1200),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_lambda": (0, 10),
    },
}

INT_PARAMS = {
    "max_depth", "min_samples_leaf", "n_estimators",
    "num_leaves", "min_child_samples",
}
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal[
    "cbr", "cbc", "lnr",
    "dtr", "dtc",
    "rfr", "rfc",
    "svr", "svc",
    "lvr", "lvc",
    "lbr", "lbc",
    "xbr", "xbc"
    ]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    @model_validator(mode="after")
    def validate_hyperparameters(self):
        params = self.hyperparameters

        if self.name in ("cbr", "cbc"):
            return self

        allowed = PARAM_RANGES.get(self.name, {})

        for key, value in params.items():
            if self.name in ("svc", "svr") and key == "gamma":
                if value not in ("scale", "auto"):
                    raise ValueError("gamma must be scale or auto")
                continue

            if key not in allowed:
                raise ValueError(
                    f"{key} is not allowed for {self.name}"
                )

            if value is None and key == "max_depth" and self.name in ("rfr", "rfc"):
                continue

            low, high = allowed[key]

            if key in INT_PARAMS:
                if type(value) is not int:
                    raise ValueError(f"{key} must be an integer")
            elif type(value) not in (int, float):
                raise ValueError(f"{key} must be numeric")

            if not low <= value <= high:
                raise ValueError(
                    f"{key} must be between {low} and {high}"
                )

        return self

class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    missing_values: Literal["native", "median", "mean", "most_frequent"]
    categorical_encoding: Literal[ "native", "onehot", "ordinal"]
    scaling: Literal["none", "standard", "robust"]


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["baseline"] = "baseline"
    drop_columns: list[str] = Field(default_factory=list)


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["run_experiment"]
    hypothesis: str = Field(min_length=1)

    model: ModelConfig
    features: FeatureConfig

    reasoning: str = Field(min_length=1)
    estimated_time_sec: int = Field(gt=0, le=10800)

class MetricResult(BaseModel):
    name: str
    value: float
    direction: Literal["maximize", "minimize"]


class ExperimentResult(BaseModel):
    experiment_id: str
    status: Literal["success", "failed"]
    model_name: str

    metric: MetricResult | None = None
    cv_std: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    duration_sec: float = Field(ge=0)
    error: str | None = None


