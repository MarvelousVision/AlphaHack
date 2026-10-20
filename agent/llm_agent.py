import json
from openai import AsyncOpenAI
from agent.schemas import AgentAction 
from agent.prompts import SYSTEM_PROMPT_TEMPLATE 

class AutoResearchAgent:
    def __init__(self, client: AsyncOpenAI, model_name: str, task_description: str):
        self.client = client
        self.model_name = model_name
        self.task_description = task_description

    async def propose(self, history: list) -> dict:
        best_exp = max(history, key=lambda x: x["score"]) if history else None
        recent_history = history[-5:] if len(history) > 5 else history
        
        history_str = json.dumps(recent_history, indent=2, ensure_ascii=False)
        best_str = json.dumps(best_exp, indent=2, ensure_ascii=False) if best_exp else "Пока нет успешных экспериментов"

        # Финальный промпт, подставляем данные
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=self.task_description,
            best_str=best_str,
            history_str=history_str
        )

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        try:
            raw_json = json.loads(response.choices[0].message.content)
            validated_action = AgentAction(**raw_json)
            return validated_action.model_dump()
        except Exception as e:
            return {
                "action": "run_experiment",
                "hypothesis": "Fallback baseline due to LLM error",
                "model": {"name": "cbr", "hyperparameters": {"iterations": 100}},
                "preprocessing": {"missing_values": "native"},
                "features": {"strategy": "baseline", "drop_columns": []},
                "reasoning": f"Error: {str(e)}",
                "estimated_time_sec": 300
            }