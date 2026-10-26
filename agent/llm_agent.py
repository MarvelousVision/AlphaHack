import json
from openai import AsyncOpenAI
from core.schemas import OptunaExperimentPlan, PARAM_RANGES, INT_PARAMS
from agent.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    STRATEGY_RECOVERY,
    STRATEGY_STAGNATION,
    STRATEGY_TIME_PRESSURE,
    LESSONS_TEMPLATE,
    ANTI_PATTERNS,
    ENSEMBLE_PROMPT,
    MODELS_REGRESSION,
    MODELS_CLASSIFICATION,
)

_MODEL_TASK_COMPAT = {
    "cbr": "regression", "lbr": "regression", "xbr": "regression",
    "rfr": "regression", "dtr": "regression", "svr": "regression",
    "lvr": "regression", "lnr": "regression",
    "cbc": "classification", "lbc": "classification", "xbc": "classification",
    "rfc": "classification", "dtc": "classification", "svc": "classification",
    "lvc": "classification",
}

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

    def _params_to_search_space(self, model_name: str, params: dict) -> dict:
        space = {}
        for k, v in params.items():
            if v is None:
                continue
            if k in INT_PARAMS:
                lo = max(1, int(v) - 2)
                space[k] = {"type": "int", "low": lo, "high": int(v) + 2}
            elif isinstance(v, float):
                space[k] = {
                    "type": "float",
                    "low": round(max(v * 0.5, 1e-5), 5),
                    "high": round(v * 2.0, 5),
                    "log": k in ("learning_rate", "C", "gamma"),
                }
        return space

    def _get_available_params_string(self) -> str:
        """Генерирует список разрешенных параметров из PARAM_RANGES."""
        lines = []
        for model_name, params in PARAM_RANGES.items():
            if not params:
                continue
            param_names = ", ".join(params.keys())
            lines.append(f"- '{model_name}': {param_names}")
        return "\n".join(lines)

    def _detect_situation(self, history: list[dict], best_result: dict | None,
                      remaining_time_sec: float, task_info: dict) -> str:
        successful = [h for h in history if h.get("status") == "success"]
        failed = [h for h in history if h.get("status") == "failed"]

        if remaining_time_sec < 1200:
            return "time_pressure"
        if failed and len(history) >= 2 and all(
            h.get("status") == "failed" for h in history[-2:]
        ):
            return "recovery"
        if self._count_stagnation(history, task_info.get("metric_name", "rmse")) >= 3:
            return "stagnation"
        if not successful or best_result is None:
            return "exploration"
        if len(successful) < 5:
            return "exploration"
        return "exploitation"

    def _count_stagnation(self, history: list[dict], metric_name: str) -> int:
        successful = [
            h for h in history
            if h.get("status") == "success" and h.get("score") is not None
        ]
        if len(successful) < 2:
            return 0

        is_lower_better = metric_name in ("rmse", "mae", "mse")

        best_so_far = None
        best_idx = 0
        for i, exp in enumerate(successful):
            s = exp["score"]
            if best_so_far is None:
                best_so_far, best_idx = s, i
            elif (s < best_so_far) if is_lower_better else (s > best_so_far):
                best_so_far, best_idx = s, i

        return len(successful) - 1 - best_idx

    def _build_lessons(self, history: list[dict], metric_name: str) -> str:
        """Извлекает уроки из истории экспериментов."""
        if not history:
            return ""

        lessons = []
        successful = [h for h in history if h.get("status") == "success"]
        failed = [h for h in history if h.get("status") == "failed"]

        is_lower_better = metric_name in ("rmse", "mae", "mse")

        #Лучшие модели
        if successful:
            by_model: dict[str, list[float]] = {}
            for exp in successful:
                model = exp.get("model", "")
                score = exp.get("score")
                if score is not None:
                    by_model.setdefault(model, []).append(score)

            if is_lower_better:
                best_models = sorted(by_model.items(), key=lambda x: min(x[1]))[:2]
            else:
                best_models = sorted(by_model.items(), key=lambda x: max(x[1]), reverse=True)[:2]
            
            if best_models:
                names = ", ".join(m for m, _ in best_models)
                lessons.append(f"Лучше всего работают: {names}. Сделай на них ставку.")

        #Провалившиеся модели
        if failed:
            failed_models = sorted(set(h.get("model", "") for h in failed))
            lessons.append(f"Избегай: {', '.join(failed_models)} — они падали с ошибками.")

        # Лучшая конфигурация
        if successful:
            if is_lower_better:
                best_exp = min(successful, key=lambda x: x.get("score") or float("inf"))
            else:
                best_exp = max(successful, key=lambda x: x.get("score") or float("-inf"))
            
            params_str = ", ".join(
            f"{k}={v}" for k, v in (best_exp.get("best_params") or {}).items()
            )
            lessons.append(
                f"Лучшая конфигурация: {best_exp['model']}({params_str}) = {best_exp['score']:.4f}. "
                f"Отталкивайся от неё."
            )

        durations = [h.get("duration_sec", 0) for h in successful if h.get("duration_sec")]
        if durations:
            avg_dur = sum(durations) / len(durations)
            lessons.append(f"Среднее время эксперимента: {avg_dur:.0f} сек.")

        return "\n".join(f"- {l}" for l in lessons)

    def _validate_plan_sanity(self, plan: dict, history: list[dict], #!!!
                              task_info: dict) -> tuple[bool, str]:
        """Проверяет адекватность плана до запуска."""

        model = plan.get("model", {}).get("name", "")
        model_obj = plan.get("model", {})
        params = model_obj.get("hyperparameters") or model_obj.get("search_space") or {}
        task_type = task_info.get("task_type", "").lower()

        # Совместимость модели и задачи
        expected_task = _MODEL_TASK_COMPAT.get(model)
        if expected_task:
            is_cls = task_type in ("binary", "multiclass", "classification")
            is_reg = task_type == "regression"
            if expected_task == "regression" and not is_reg:
                raise ValueError(f"Модель {model} для регрессии, а задача — {task_type}")
            if expected_task == "classification" and not is_cls:
                raise ValueError(f"Модель {model} для классификации, а задача — {task_type}")

        #SVM на больших данных
        n_rows = task_info.get("n_rows")
        if model in ("svr", "svc") and isinstance(n_rows, int) and n_rows >= 10000:
            raise ValueError(f"SVM на {n_rows} строках = OOM")

        for exp in history:
            exp_params = exp.get("best_params") or exp.get("hyperparameters") or {}
            if exp.get("model") == model and exp_params == params:
                raise ValueError("Эта конфигурация уже была в истории")

        return True, "OK"
    

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

        situation = self._detect_situation(history, best_result, remaining_time_sec, task_info)
        lessons_text = self._build_lessons(history, task_info.get("metric_name", "rmse"))
        lessons_block = LESSONS_TEMPLATE.format(lessons=lessons_text) if lessons_text else ""

        situation_block = ""
        if situation == "recovery":
            errors = [h for h in history if h.get("status") == "failed"][-3:]
            errors_summary = "\n".join(
                f"  - {e.get('model')}: {e.get('error', 'unknown')}" for e in errors
            )
            situation_block = "\n\nПРИОРИТЕТНАЯ СТРАТЕГИЯ (переопределяет базовую):\n" + STRATEGY_RECOVERY.format(errors_summary=errors_summary)
        elif situation == "stagnation":
            count = self._count_stagnation(history, task_info.get("metric_name", "rmse"))
            situation_block = "\n\nПРИОРИТЕТНАЯ СТРАТЕГИЯ (переопределяет базовую):\n" + STRATEGY_STAGNATION.format(stagnation_count=count)
        elif situation == "time_pressure":
            situation_block = "\n\nПРИОРИТЕТНАЯ СТРАТЕГИЯ (переопределяет базовую):\n" + STRATEGY_TIME_PRESSURE

        anti_patterns_block = "\n\n" + ANTI_PATTERNS

        lessons_final_block = "\n\n" + lessons_block if lessons_block else ""
        
        system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(
            task_type=task_info.get("task_type", "unknown"),
            metric_name=task_info.get("metric_name", "unknown"),
            task_description=task_info.get("description", ""),
            n_rows=task_info.get("n_rows", "unknown"),
            remaining_time_sec=remaining_time_sec,
            available_params=available_params_str,
            best_str=best_str,
            history_str=history_str
            )
            + situation_block
            + anti_patterns_block
            + lessons_final_block
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
            
            validated_action = OptunaExperimentPlan(**raw_json)
            plan_dict = validated_action.model_dump()

            ok, msg = self._validate_plan_sanity(plan_dict, history, task_info)
            if not ok:
                print(f"[Agent Warning] Plan rejected: {msg}. Retrying...")
                
            return plan_dict
            
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
                params_str = str(sorted((exp.get("best_params") or {}).items()))
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
                best_params = dict(
                    best_exp.get("best_params") or best_exp.get("hyperparameters") or {}
                )

                space = self._params_to_search_space(best_model_name, best_params)
                if space:
                    fallback_dict = {
                        "action": "run_optuna",
                        "hypothesis": f"Smart Fallback: optuna around best ({best_model_name})",
                        "model": {"name": best_model_name, "search_space": space},
                        "reasoning": "LLM failed. Narrow Optuna search around best known config.",
                        "estimated_time_sec": 600,
                    }

            if fallback_dict is None:
                fallbacks = self._get_fallback_configs(is_classification)

                for cfg in fallbacks:
                    space = self._params_to_search_space(cfg["name"], cfg["hyperparameters"])
                    if not space:
                        continue
                    fallback_dict = {
                        "action": "run_optuna",
                        "hypothesis": f"Safe Fallback: optuna on {cfg['name']}",
                        "model": {"name": cfg["name"], "search_space": space},
                        "reasoning": "LLM failed. Optuna search from a robust baseline.",
                        "estimated_time_sec": 600,
                    }
                    break

                if fallback_dict is None:
                    fallback_dict = {
                        "action": "run_optuna",
                        "hypothesis": "Подобрать глубину и learning rate CatBoost",
                        "model": {
                            "name": "cbr",
                            "search_space": {
                            "depth": {
                                "type": "int",
                                "low": 4,
                                "high": 9
                            },
                            "learning_rate": {
                                "type": "float",
                                "low": 0.02,
                                "high": 0.15,
                                "log": True
                            }
                            }
                        },
                        "reasoning": "Попробовать уменьшить RMSE",
                        "estimated_time_sec": 900
                    }


            try:
                return OptunaExperimentPlan(**fallback_dict).model_dump()
            except Exception as ve:
                raise RuntimeError(f"Fallback validation failed: {ve}")
