"""Собрать плоский ZIP для Boosters, не меняя исходники команды."""

from pathlib import Path
import ast
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build_boosters"
ZIP_PATH = ROOT / "solution_boosters.zip"

# Только модули, необходимые решению; тесты и baseline.py исключены.
SOURCES = [
    "solution.py",
    "core/schemas.py",
    "core/experiment_runner.py",
    "core/experiment_tracker.py",
    "core/orchestrator.py",
    "core/ml_adapter.py",
    "agent/llm_agent.py",
    "agent/prompts.py",
    "data_tools/models.py",
    "data_tools/preprocessing.py",
    "data_tools/models2.py",
    "requirements.txt",
]


def flatten_imports(code: str, filename: str) -> str:
    # Не трогаем сами исходные модули: заменяем импорты только в сборке.
    code = re.sub(
        r"(?m)^(\s*from\s+)(?:core|agent|data_tools)\.([A-Za-z_]\w*)(\s+import\s+)",
        r"\1\2\3",
        code,
    )

    tree = ast.parse(code, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in {"core", "agent", "data_tools"} for alias in node.names):
                raise ValueError(f"Unsupported package import in {filename}, line {node.lineno}: adjust builder")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").split(".")[0] in {"core", "agent", "data_tools"}:
                raise ValueError(f"Unsupported package import in {filename}, line {node.lineno}: adjust builder")
    return code


def main() -> None:
    # На старте не удаляем существующую сборку: сначала проверим исходники.
    selected = []
    for relpath in SOURCES:
        src = ROOT / relpath
        if not src.is_file():
            if relpath == "core/ml_adapter.py":  # необязательный модуль
                continue
            raise FileNotFoundError(f"Missing required file: {src}")
        selected.append((src, Path(relpath).name))

    names = [name for _, name in selected]
    if len(names) != len(set(names)):
        raise ValueError("Filename collision after flattening")

    BUILD.mkdir(exist_ok=True)
    for old in BUILD.iterdir():
        if old.is_file():
            old.unlink()

    for src, name in selected:
        dst = BUILD / name
        if src.suffix == ".py":
            text = flatten_imports(src.read_text(encoding="utf-8-sig"), str(src))
            dst.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(src, dst)

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for _, name in selected:
            zf.write(BUILD / name, arcname=name)

    print(f"Created: {ZIP_PATH}")
    print("Files:", ", ".join(names))
    print("Check imports and run from build_boosters before uploading to Boosters.")


if __name__ == "__main__":
    main()
