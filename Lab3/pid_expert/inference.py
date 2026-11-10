"""Механизм прямого вывода для экспертной системы PX4 PID."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Rule:
    id: str
    conditions: list[tuple[str, str, Any]]
    conclusion_type: str          # recommendation | refusal | question | final_question
    conclusion_text: str
    source: str
    priority: int
    ask_variable: str | None = None


@dataclass
class FiredRule:
    rule: Rule
    matched_facts: list[tuple[str, Any]]


@dataclass
class KnowledgeBase:
    variables: dict[str, dict]
    rules: list[Rule]


def _eval_condition(fact: Any, op: str, expected: Any) -> bool:
    if op == "=":
        return fact == expected
    if op == "!=":
        return fact != expected
    if op == "in":
        return fact in expected
    raise ValueError(f"Неизвестный оператор: {op}")


def load_knowledge_base(path: str | Path) -> KnowledgeBase:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    variables = raw["variables"]
    rules: list[Rule] = []

    for item in raw["rules"]:
        conditions = [(c[0], c[1], c[2]) for c in item["if"]]
        then = item["then"]
        rules.append(
            Rule(
                id=item["id"],
                conditions=conditions,
                conclusion_type=then["type"],
                conclusion_text=then["text"],
                source=item["source"],
                priority=int(item["priority"]),
                ask_variable=then.get("ask_variable"),
            )
        )

    known_vars = set(variables.keys())
    for r in rules:
        for var, _, _ in r.conditions:
            if var not in known_vars:
                raise ValueError(f"Правило {r.id} ссылается на неизвестную переменную '{var}'")
        if r.ask_variable and r.ask_variable not in known_vars:
            raise ValueError(f"Правило {r.id}: ask_variable '{r.ask_variable}' не объявлена")

    return KnowledgeBase(variables=variables, rules=rules)


class InferenceSession:
    """Факты текущего сеанса хранятся отдельно от правил."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.facts: dict[str, Any] = {}
        self.fired: list[FiredRule] = []
        self.decisions: list[FiredRule] = []

    def has(self, var: str) -> bool:
        return var in self.facts

    def get(self, var: str) -> Any | None:
        return self.facts.get(var)

    def set_fact(self, var: str, value: Any) -> None:
        self.facts[var] = value

    def unknown_variables(self) -> list[str]:
        return [v for v in self.kb.variables if v not in self.facts]

    # ---------- проверка применимости ----------

    def _rule_matches(self, rule: Rule) -> bool:
        for var, op, expected in rule.conditions:
            if var not in self.facts:
                return False
            if not _eval_condition(self.facts[var], op, expected):
                return False
        return True

    def applicable_rules(self) -> list[Rule]:
        """Все ещё не сработавшие правила, условия которых выполнены."""
        fired_ids = {f.rule.id for f in self.decisions}
        matches = [
            r for r in self.kb.rules
            if r.id not in fired_ids and self._rule_matches(r)
        ]
        # Сортировка по приоритету — разрешение конфликтов
        matches.sort(key=lambda r: -r.priority)
        return matches

    # ---------- основной цикл ----------

    def run_inference(self) -> FiredRule | None:
        """
        Выполняет вывод до тех пор, пока не будет получено терминальное
        правило (refusal / recommendation / question / final_question)
        либо пока не исчерпаются применимые правила.

        Терминальные правила НЕ удаляются из decisions — это позволяет
        Dialogue объяснить, почему система задала текущий вопрос.
        """
        while True:
            applicable = self.applicable_rules()
            if not applicable:
                break

            chosen = applicable[0]
            matched = [(v, self.facts[v]) for v, _, _ in chosen.conditions]
            fired = FiredRule(rule=chosen, matched_facts=matched)
            self.fired.append(fired)
            self.decisions.append(fired)

            if chosen.conclusion_type == "refusal":
                return fired
            if chosen.conclusion_type in ("recommendation",
                                          "question",
                                          "final_question"):
                return fired

        return self.decisions[-1] if self.decisions else None

    def last_terminal(self) -> FiredRule | None:
        for d in reversed(self.decisions):
            if d.rule.conclusion_type in ("refusal", "recommendation",
                                          "question", "final_question"):
                return d
        return None
