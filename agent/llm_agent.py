import json
from openai import AsyncOpenAI
from core.schemas import ExperimentPlan
from agent.prompts import SYSTEM_PROMPT_TEMPLATE 

class AutoResearchAgent:
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    async def propose(
    self,
    history: list[dict],
    best_result: dict | None,
    task_info: dict,
    remaining_time_sec: float,
    ) -> dict:
            
        recent_history = history[-5:] if len(history) > 5 else history
        history_str = json.dumps(recent_history, indent=2, ensure_ascii=False)
        best_str = json.dumps(best_result, indent=2, ensure_ascii=False) if best_result else "Пока нет успешных экспериментов"

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_type=task_info.get("task_type", "unknown"),
            metric_name=task_info.get("metric_name", "unknown"),
            task_description=task_info.get("description", ""),
            remaining_time_sec=remaining_time_sec,
            best_str=best_str,
            history_str=history_str
        )

        try:

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            raw_json = json.loads(response.choices[0].message.content)
            
            validated_action = ExperimentPlan(**raw_json)
            return validated_action.model_dump()
            
        except Exception as e:
            print(f"[Agent Warning] LLM or API Error: {e}. Using fallback.")
            
            is_classification = task_info.get("task_type", "").lower() in (
            "binary",
            "multiclass",
            "classification",
            )

            fallback_model = "cbc" if is_classification else "cbr"

            fallback_dict = {
                "action": "run_experiment",
                "hypothesis": f"Fallback baseline due to error: {str(e)[:50]}",
                "model": {
                    "name": fallback_model, 
                    "hyperparameters": {"iterations": 100}
                },
                "preprocessing": {
                    "missing_values": "native", 
                    "categorical_encoding": "native", 
                    "scaling": "none"
                },
                "features": {
                    "strategy": "baseline", 
                    "drop_columns": []
                },
                "reasoning": "Fallback triggered to ensure pipeline doesn't crash.",
                "estimated_time_sec": 300
            }

            try:
                validated_fallback = ExperimentPlan(**fallback_dict)
                return validated_fallback.model_dump()
            
            except Exception as validation_error:
                print(f"[Agent CRITICAL] Fallback validation failed: {validation_error}")
                return fallback_dict