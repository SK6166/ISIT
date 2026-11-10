"""Модуль объяснения: строит трассировку по фактически сработавшим правилам."""

from __future__ import annotations

from inference import FiredRule, InferenceSession


def render_explanation(fired: FiredRule, session: InferenceSession) -> str:
    rule = fired.rule
    lines: list[str] = []

    lines.append(f"[Объяснение правила {rule.id}]")
    lines.append("Условия, которые выполнились:")
    for var, value in fired.matched_facts:
        lines.append(f"  • {var} = {value}")

    lines.append("")
    lines.append(f"Вывод ({rule.conclusion_type}): {rule.conclusion_text}")
    lines.append(f"Источник знания: {rule.source}")
    lines.append(f"Приоритет правила: {rule.priority}")

    # Показываем всю цепочку, если правил несколько
    if len(session.decisions) > 1:
        lines.append("")
        lines.append("Полная цепочка сработавших правил:")
        for i, d in enumerate(session.decisions, 1):
            facts_str = ", ".join(f"{v}={val}" for v, val in d.matched_facts)
            lines.append(f"  {i}. {d.rule.id}: [{facts_str}] → {d.rule.conclusion_text}")

    return "\n".join(lines)
