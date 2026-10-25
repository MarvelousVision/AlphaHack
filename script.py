"""Локальная проверка AutoResearchAgent.propose_ensemble.

Запуск из корня проекта:
    python test_propose_ensemble.py

Не требует OPENAI_API_KEY: propose_ensemble не обращается к LLM.
"""

import asyncio
import sys
from pathlib import Path

# agent/llm_agent.py использует плоские импорты (`from schemas import ...`,
# `from prompts import ...`), которые работают после build_zip. Для локального
# запуска без сборки подкладываем нужные каталоги в sys.path.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "agent"))

from llm_agent import AutoResearchAgent          # noqa: E402
from schemas import EnsemblePlan                  # noqa: E402


EXPECTED = {
    "regression": ["cbr", "lbr", "rfr"],
    "binary":     ["cbc", "lbc", "rfc"],
    "multiclass": ["cbc", "lbc", "rfc"],
}


def make_agent() -> AutoResearchAgent:
    # client=None гарантирует: если propose_ensemble полезет в LLM,
    # мы получим AttributeError и сразу узнаем о нарушении контракта.
    return AutoResearchAgent(client=None, model_name="unused")


async def check_valid(task_type: str) -> None:
    print(f"\n=== propose_ensemble: task_type={task_type} ===")
    info = {"task_type": task_type, "metric_name": "any", "n_rows": 87670}

    plan_dict = await make_agent().propose_ensemble(info)
    print("raw:", plan_dict)

    # 1. Структура ответа
    assert isinstance(plan_dict, dict), "propose_ensemble must return dict"
    assert set(plan_dict) == {"action", "hypothesis", "models", "reasoning"}, (
        f"unexpected keys: {set(plan_dict)}"
    )
    assert plan_dict["action"] == "run_ensemble"
    assert isinstance(plan_dict["hypothesis"], str) and plan_dict["hypothesis"]
    assert isinstance(plan_dict["reasoning"], str) and plan_dict["reasoning"]

    # 2. Состав моделей (list или tuple — model_dump отдаёт tuple)
    models = tuple(plan_dict["models"])
    assert models == tuple(EXPECTED[task_type]), (
        f"expected {EXPECTED[task_type]}, got {models}"
    )
    assert len(set(models)) == 3, "models must be unique"

    # 3. Round-trip через EnsemblePlan — ровно то, что делает
    #    Orchestrator.propose_ensemble в solution.py
    plan = EnsemblePlan.model_validate(plan_dict)
    plan.validate_task(task_type)

    print(f"OK: {models}")


async def check_invalid_task_type() -> None:
    print("\n=== propose_ensemble: unknown task_type is rejected ===")
    try:
        await make_agent().propose_ensemble({"task_type": "unknown"})
    except ValueError as exc:
        print(f"OK, ValueError: {exc}")
    else:
        raise AssertionError("expected ValueError for unknown task_type")


async def check_deterministic() -> None:
    print("\n=== determinism ===")
    info = {"task_type": "multiclass"}
    agent = make_agent()
    first = await agent.propose_ensemble(info)
    second = await agent.propose_ensemble(info)
    assert first == second, f"non-deterministic: {first} != {second}"
    print("OK: two consecutive calls yield identical plans")


async def check_no_llm_dependency() -> None:
    print("\n=== no LLM calls ===")
    # Если бы метод обращался к self.client, client=None дал бы AttributeError.
    # Здесь же явно просим хотя бы одну успешную проверку с None-клиентом.
    await make_agent().propose_ensemble({"task_type": "regression"})
    print("OK: works with client=None (no hidden LLM dependency)")


async def main() -> None:
    for task_type in ("regression", "binary", "multiclass"):
        await check_valid(task_type)
    await check_invalid_task_type()
    await check_deterministic()
    await check_no_llm_dependency()
    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    asyncio.run(main())
