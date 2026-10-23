import time
import json
import sys
from pathlib import Path

from data_tools.preprocessing import prepare_data

t0 = time.perf_counter()
X_train, y, X_test, numeric, cat, task, ids, tgt, id_name = prepare_data('data/task1')
print(f"Preprocessing: {time.perf_counter() - t0:.1f}s")
print(f"X_train shape: {X_train.shape}")
print(f"task: {task}, metric: {'RMSE' if task == 'regression' else 'F1'}")

sys.modules["solution"] = sys.modules[__name__]
globals().update({
    "X_train": X_train.to_pandas(), "y": y.to_pandas(),
    "X_test": X_test.to_pandas(), "ids": ids.to_pandas(),
    "numeric": numeric, "cat": cat, "task": task,
    "tgt": tgt, "id_name": id_name,
    "METRIC": "RMSE" if task == "regression" else "f1_macro",
    "WAY_TO_JSON": "artifacts/bench_experiments.json",
})
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/bench_experiments.json").write_text("{}")

from data_tools.models import unimodel

ITER_GRID = [100, 300, 500, 800, 1200]
DEPTH_GRID = [4, 6, 8]

print(f"\n{'='*85}")
print(f"{'depth':>6} | {'iterations':>10} | {'RMSE':>10} | {'±std':>8} | {'min':>8} | {'max':>8} | {'time':>8}")
print(f"{'-'*85}")

results_table = []
for depth in DEPTH_GRID:
    for iters in ITER_GRID:
        t0 = time.perf_counter()
        try:
            unimodel(
                model="cbr",
                iterations=iters,
                depth=depth,
                learning_rate=0.05,
                random_seed=42,
                verbose=False,
            )
        except Exception as e:
            print(f"{depth:>6} | {iters:>10} | ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0

        results = json.loads(Path("artifacts/bench_experiments.json").read_text())
        if not results:
            print(f"{depth:>6} | {iters:>10} | ERROR: no json")
            continue
        last_key = max(results.keys(), key=int)
        rec = results[last_key]

        row = (depth, iters, rec["mean_score"], rec["score_std"], elapsed)
        results_table.append(row)
        print(f"{depth:>6} | {iters:>10} | {row[2]:>10.2f} | {row[3]:>8.2f} | "
              f"{rec['min_score']:>8.2f} | {rec['max_score']:>8.2f} | {elapsed:>6.1f}s")

print(f"{'='*85}")

# Правильный поиск лучшего: mean на индексе 2
best = min(results_table, key=lambda x: x[2])
print(f"\n🏆 Лучший RMSE: {best[2]:.2f} при depth={best[0]}, iter={best[1]} ({best[4]:.1f}s)")

# Анализ плато — сравнение только внутри одной depth
print("\n📈 Анализ плато (внутри каждой depth):")
for depth in DEPTH_GRID:
    group = [r for r in results_table if r[0] == depth]
    print(f"\n  depth={depth}:")
    for i in range(1, len(group)):
        prev = group[i-1]
        curr = group[i]
        delta = (prev[2] - curr[2]) / prev[2] * 100
        marker = " ⬅ плато" if 0 <= delta < 1.0 else ""
        print(f"    {prev[1]:>4} → {curr[1]:<4} | RMSE: {prev[2]:>6.2f} → {curr[2]:>6.2f} | "
              f"улучшение: {delta:>+5.1f}%{marker}")

# Сводка: лучший по глубине
print("\n📊 Лучший RMSE для каждой depth:")
for depth in DEPTH_GRID:
    group = [r for r in results_table if r[0] == depth]
    if not group:
        continue
    b = min(group, key=lambda x: x[2])
    print(f"  depth={depth}: RMSE={b[2]:.2f} при iter={b[1]}, time={b[4]:.1f}s")