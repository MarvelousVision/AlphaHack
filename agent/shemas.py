from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ModelConfig(BaseModel):
    name: str = Field(description="Должно быть строго 'lnr', 'cbr' или 'cbc'")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

class PreprocessingConfig(BaseModel):
    missing_values: str = "native"
    categorical_encoding: str = "native"
    scaling: str = "none"

class FeaturesConfig(BaseModel):
    strategy: str = "baseline"
    drop_columns: List[str] = Field(default_factory=list)

class AgentAction(BaseModel):
    action: str = "run_experiment"
    hypothesis: str
    model: ModelConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    reasoning: str
    estimated_time_sec: int