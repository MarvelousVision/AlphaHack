import json
from openai import AsyncOpenAI
from schemas import ExperimentPlan, EnsemblePlan, PARAM_RANGES, INT_PARAMS
from prompts import SYSTEM_PROMPT_TEMPLATE

REGRESSION_MODELS_TEXT = "cbr, lbr, xbr, rfr, dtr, svr, lvr, lnr"
CLASSIFICATION_MODELS_TEXT = "cbc, lbc, xbc, rfc, dtc, svc, lvc"

REGRESSION_FALLBACK = ["cbr", "lbr", "rfr"]
CLASSIFICATION_FALLBACK = ["cbc", "lbc", "rfc"]

class AutoResearchAgent:
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def _get_fallback_configs(self, is_classification: bool):
        # конфиги если qwen недоступен
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

    def _check_bounds(self, model_name: str, params: dict) -> dict:
        clamped = dict(params)
        
        bounds = PARAM_RANGES.get(model_name, {})

        for param, value in clamped.items():
            if value is None:
                continue
                
            if param in bounds:
                min_val, max_val = bounds[param]
                
                if param in INT_PARAMS or isinstance(value, int):
                    clamped[param] = max(min_val, min(int(value), max_val))
                elif isinstance(value, float):
                    clamped[param] = round(max(min_val, min(value, max_val)), 4)
                    
        return clamped

    def _get_available_params_string(self) -> str:
        """Генерирует список разрешенных параметров из PARAM_RANGES."""
        lines = []
        for model_name, params in PARAM_RANGES.items():
            if not params:
                continue
            param_names = ", ".join(params.keys())
            lines.append(f"- '{model_name}': {param_names}")
        return "\n".join(lines)

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
        available_params_str = self._get_available_params_string()

        
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_type=task_info.get("task_type", "unknown"),
            metric_name=task_info.get("metric_name", "unknown"),
            task_description=task_info.get("description", ""),
            n_rows=task_info.get("n_rows", "unknown"),
            remaining_time_sec=remaining_time_sec,
            available_params=available_params_str,
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

            if isinstance(raw_json, list):
                if len(raw_json) > 0 and isinstance(raw_json[0], dict):
                    raw_json = raw_json[0]
                else:
                    raise ValueError(f"LLM returned an empty or invalid list: {raw_json}")

            if not isinstance(raw_json, dict):
                raise ValueError(f"LLM returned an invalid format (expected dict, got {type(raw_json)}): {raw_json}")
            
            validated_action = ExperimentPlan(**raw_json)
            return validated_action.model_dump()
            
        except Exception as e:
            print(f"[Agent Warning] LLM or API Error: {e}. Using fallback.")
            
            is_classification = task_info.get("task_type", "").lower() in (
            "binary",
            "multiclass",
            "classification",
            )

            past_configs = set()
            for exp in history:
                # Не повторяем что уже было
                model_str = exp.get("model", "")
                params_str = str(sorted(exp.get("hyperparameters", {}).items()))
                past_configs.add(f"{model_str}_{params_str}")

            def is_duplicate(model_name, params):
                model_str = model_name
                params_str = str(sorted(params.items()))
                return f"{model_str}_{params_str}" in past_configs

            successful_exps = [exp for exp in history if exp.get("status") == "success"]
            
            fallback_dict = None
            
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
                for attempt in range(5): #пусть 5, если на хватит, сделаем больше
                    mutated_params = dict(best_params)
                    
                    if "learning_rate" in mutated_params:
                        mutated_params["learning_rate"] = round(mutated_params["learning_rate"] * (0.8 - attempt * 0.1), 4)
                    
                    if "iterations" in mutated_params:
                        mutated_params["iterations"] = int(mutated_params["iterations"] * (1.5 + attempt * 0.5))
                    elif "n_estimators" in mutated_params:
                        mutated_params["n_estimators"] = int(mutated_params["n_estimators"] * (1.5 + attempt * 0.5))
                    
                    if "max_depth" in mutated_params and mutated_params["max_depth"] is not None:
                        mutated_params["max_depth"] = int(mutated_params["max_depth"] * (1.2 + attempt * 0.1))
                    
                    if "C" in mutated_params: #Для SVM
                        mutated_params["C"] = round(mutated_params["C"] * (1.5 + attempt), 2)

                    #Если параметров нет вообще добавим дефолтный lr
                    if not mutated_params and best_model_name not in ["lnr"]:
                         mutated_params["learning_rate"] = 0.05

                    mutated_params = self._check_bounds(best_model_name, mutated_params)

                    if not is_duplicate(best_model_name, mutated_params):
                        fallback_dict = {
                            "action": "run_experiment",
                            "hypothesis": f"Smart Fallback: tuning best model ({best_model_name})",
                            "model": {"name": best_model_name, "hyperparameters": mutated_params},
                            "features": {"strategy": "baseline", "drop_columns": []},
                            "reasoning": "LLM failed. Mutating best known config to find a unique untested combination.",
                            "estimated_time_sec": 300
                        }
                        break #уникальный конфиг найден, выходим
                
                #Если есть совпадения после 5 попытки -> сц. 2
                if fallback_dict is None:
                    print("[Agent Warning] Mutation failed to find unique config. Falling back to defaults.")

            if fallback_dict is None:
                # Сценарий 2: Истории нет или дубликат, берем заранее подготовленный кофиг выше
                fallbacks = self._get_fallback_configs(is_classification)
                
                for cfg in fallbacks:
                    if not is_duplicate(cfg["name"], cfg["hyperparameters"]):
                        fallback_dict = {
                            "action": "run_experiment",
                            "hypothesis": f"Safe Fallback: exploring {cfg['name']}",
                            "model": cfg,
                            "features": {"strategy": "baseline", "drop_columns": []},
                            "reasoning": "LLM failed. Trying a robust baseline from a predefined list.",
                            "estimated_time_sec": 300
                        }
                        break

                if fallback_dict is None:
                    fallback_dict = {
                        "action": "run_experiment",
                        "hypothesis": "Emergency Fallback: extreme config",
                        "model": {
                            "name": "lbc" if is_classification else "lbr", 
                            "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1}
                        },
                        "features": {"strategy": "baseline", "drop_columns": []},
                        "reasoning": "All fallbacks tested. Using ultra-fast baseline.",
                        "estimated_time_sec": 300
                    }

            try:
                return ExperimentPlan(**fallback_dict).model_dump()
            except Exception as ve:
                raise RuntimeError(f"Fallback validation failed: {ve}")

    async def propose_ensemble(self, task_info: dict) -> dict:
        task_type = str(task_info.get("task_type", "")).lower()

        if task_type == "regression":
            models = list(REGRESSION_FALLBACK)
        else:
            models = list(CLASSIFICATION_FALLBACK)

        plan = EnsemblePlan(
            action="run_ensemble",
            hypothesis="Three complementary classifiers",
            models=tuple(models),
            reasoning=("Combine boosting and bagging"),
        )

        plan.validate_task(task_type)

        return plan.model_dump()

    