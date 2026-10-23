import json
from openai import AsyncOpenAI
from core.schemas import ExperimentPlan
from agent.prompts import SYSTEM_PROMPT_TEMPLATE 

class AutoResearchAgent:
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def _get_fallback_configs(self, is_classification: bool):
        # Пункт 3: Заранее подготовленные конфиги (Exploration)
        if is_classification:
            return [
                {"name": "lbc", "hyperparameters": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05}},
                {"name": "cbc", "hyperparameters": {"iterations": 300, "depth": 6, "learning_rate": 0.05}},
                {"name": "xbc", "hyperparameters": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}}
            ]
        else:
            return [
                {"name": "lbr", "hyperparameters": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05}},
                {"name": "cbr", "hyperparameters": {"iterations": 300, "depth": 6, "learning_rate": 0.05}},
                {"name": "xbr", "hyperparameters": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}}
            ]

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

            successful_exps = [exp for exp in history if exp.get("status") == "success"]
            
            if successful_exps:
                # Сценарий 1: Есть история. Берем лучший и мутируем.
                metric_name = task_info.get("metric_name", "rmse")
                if metric_name in ["rmse", "mae", "mse"]:
                    best_exp = min(successful_exps, key=lambda x: x["score"] if x["score"] is not None else float('inf'))
                else:
                    best_exp = max(successful_exps, key=lambda x: x["score"] if x["score"] is not None else float('-inf'))
                
                best_model_name = best_exp["model"]
                best_params = dict(best_exp["hyperparameters"])
                
                # Мутируем параметры
                if "learning_rate" in best_params:
                    best_params["learning_rate"] = round(best_params["learning_rate"] * 0.8, 4)
                elif "iterations" in best_params:
                    best_params["iterations"] = int(best_params["iterations"] * 1.5)
                elif "n_estimators" in best_params:
                    best_params["n_estimators"] = int(best_params["n_estimators"] * 1.5)
                    
                fallback_dict = {
                    "action": "run_experiment",
                    "hypothesis": f"Smart Fallback: continuing tuning of the best model ({best_model_name})",
                    "model": {"name": best_model_name, "hyperparameters": best_params},
                    "features": {"strategy": "baseline", "drop_columns": []},
                    "reasoning": "LLM failed. Taking the best known model and slightly altering its hyperparameters.",
                    "estimated_time_sec": 300
                }
            else:
                # Сценарий 2: Истории нет, берем заранее подготовленный кофиг выше
                fallbacks = self._get_fallback_configs(is_classification)
                tried_models = [exp["model"] for exp in history]
                
                chosen_config = fallbacks[0]
                for cfg in fallbacks:
                    if cfg["name"] not in tried_models:
                        chosen_config = cfg
                        break

                fallback_dict = {
                    "action": "run_experiment",
                    "hypothesis": f"Safe Fallback: exploring {chosen_config['name']}",
                    "model": chosen_config,
                    "features": {"strategy": "baseline", "drop_columns": []},
                    "reasoning": "LLM failed. Trying a robust baseline from a predefined list.",
                    "estimated_time_sec": 300
                }

            try:
                return ExperimentPlan(**fallback_dict).model_dump()
            except Exception as ve:
                raise RuntimeError(f"Fallback validation failed: {ve}")