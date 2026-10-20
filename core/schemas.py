
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal[
        "catboost",
        "lightgbm",
        "xgboost",
        "linear_regression",
        "logistic_regression"
    ]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

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
    preprocessing: PreprocessingConfig
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


