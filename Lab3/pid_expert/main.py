"""Точка входа экспертной системы PX4 PID."""

from pathlib import Path

from dialogue import Dialogue
from inference import load_knowledge_base


def main() -> None:
    kb_path = Path(__file__).parent / "rules.yaml"
    if not kb_path.exists():
        raise SystemExit(f"Файл базы правил не найден: {kb_path}")

    kb = load_knowledge_base(kb_path)
    print(f"[Загружено правил: {len(kb.rules)}, переменных: {len(kb.variables)}]")

    Dialogue(kb).run()


if __name__ == "__main__":
    main()
