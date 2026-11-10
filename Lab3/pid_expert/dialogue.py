"""Консольный диалог экспертной системы."""

from __future__ import annotations

from inference import InferenceSession, KnowledgeBase, FiredRule


class Dialogue:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.session = InferenceSession(kb)
        # Правила, которые «попросили уточнение» и не должны
        # повторно выводиться как финальные
        self._exhausted_questions: set[str] = set()

    # ---------- ввод ----------

    def _ask_variable(self, var: str) -> None:
        info = self.kb.variables[var]
        question = info["question"]
        options = info["options"]

        print(f"\n[Вопрос] {question}")
        print(f"  Допустимые значения: {', '.join(options)}")

        # Словарь: нижний регистр -> каноническое значение
        canonical = {opt.lower(): opt for opt in options}

        while True:
            answer = input("> ").strip()
            low = answer.lower()

            if low in ("почему", "почему?", "why", "why?"):
                self._explain_current(var)
                continue
            if low in ("выход", "exit", "quit"):
                raise KeyboardInterrupt

            if low in canonical:
                self.session.set_fact(var, canonical[low])
                return

            print(f"  Некорректный ввод. Введите одно из: {', '.join(options)}")
            print("  (или 'почему?' для объяснения текущего вопроса)")

    # ---------- объяснение ----------

    def _explain_current(self, current_var: str) -> None:
        """
        Объясняет, ПОЧЕМУ задаётся текущий вопрос:
        ищем правило, которое требует эту переменную и условия которого
        уже выполнены.
        """
        # 1) Если есть терминальное правило, требующее current_var — ссылаемся на него
        for d in reversed(self.session.decisions):
            if d.rule.ask_variable == current_var:
                print()
                print(f"[Объяснение] Вопрос задан по правилу {d.rule.id} "
                      f"(приоритет {d.rule.priority})")
                print(f"  Текст правила: {d.rule.conclusion_text}")
                print(f"  Источник: {d.rule.source}")
                if d.matched_facts:
                    print("  Уже выполненные условия:")
                    for v, val in d.matched_facts:
                        print(f"    • {v} = {val}")
                return

        # 2) Иначе — ищем ещё не сработавшее правило с высоким приоритетом,
        #    которому не хватает именно current_var
        candidates = []
        for rule in self.kb.rules:
            if rule.id in {d.rule.id for d in self.session.decisions}:
                continue
            needed = [v for v, _, _ in rule.conditions if v not in self.session.facts]
            if current_var in needed:
                candidates.append(rule)

        if candidates:
            candidates.sort(key=lambda r: -r.priority)
            r = candidates[0]
            print()
            print(f"[Объяснение] Вопрос нужен, чтобы проверить правило {r.id} "
                  f"(приоритет {r.priority})")
            print(f"  Условия правила:")
            for v, op, exp in r.conditions:
                mark = "✓" if v in self.session.facts else "?"
                print(f"    [{mark}] {v} {op} {exp}")
            print(f"  Источник: {r.source}")
            return

        print("\n[Объяснение] Пока нет сработавших правил — задавайте вопросы дальше.")

    # ---------- основной цикл ----------

    def run(self) -> None:
        print("=" * 62)
        print("  Экспертная система: диагностика PID-параметров PX4")
        print("=" * 62)
        print("Отвечайте на вопросы. 'почему?' — объяснение текущего вопроса.")
        print("'выход' — завершить сеанс.\n")

        try:
            while True:
                # 1. Есть ли уже терминальное правило?
                terminal = self._find_terminal()

                if terminal is not None:
                    if terminal.rule.conclusion_type == "refusal":
                        self._print_final(terminal)
                        return

                    if terminal.rule.conclusion_type == "recommendation":
                        # Пока есть незаданные вопросы, которые могут
                        # повысить приоритет рекомендации, — задаём их
                        if self._can_refine():
                            nxt = self._pick_next_variable()
                            if nxt:
                                self._ask_variable(nxt)
                                continue
                        self._print_final(terminal)
                        return

                    # question / final_question
                    target = terminal.rule.ask_variable
                    if target and target not in self.session.facts:
                        # Задаём уточняющий вопрос
                        print(f"\n[Уточнение] {terminal.rule.conclusion_text}")
                        self._ask_variable(target)
                        continue
                    else:
                        # Переменная уже известна — этот «вопрос»
                        # больше не должен считаться финалом.
                        self._exhausted_questions.add(terminal.rule.id)
                        continue

                # 2. Терминального нет — выбираем следующий вопрос
                nxt = self._pick_next_variable()
                if nxt is None:
                    print("\n[Результат] Недостаточно данных, и ни одно правило не сработало.")
                    print("Возможно, случай выходит за границы базы знаний. "
                          "Рекомендуется Autotune.")
                    return
                self._ask_variable(nxt)

        except (KeyboardInterrupt, EOFError):
            print("\n\n[Сеанс прерван пользователем]")

    # ---------- вспомогательные ----------

    def _find_terminal(self) -> FiredRule | None:
        """
        Возвращает последнее терминальное правило, которое ещё не
        было «исчерпано» как вопрос.
        """
        for d in reversed(self.session.decisions):
            if d.rule.id in self._exhausted_questions:
                continue
            if d.rule.conclusion_type in (
                "refusal", "recommendation", "question", "final_question"
            ):
                return d
        return None

    def _can_refine(self) -> bool:
        """Есть ли ещё переменные, которые могут уточнить диагноз?"""
        return bool(self._pick_next_variable())

    def _pick_next_variable(self) -> str | None:
        """
        Выбирает переменную, которая разблокирует правило с максимальным
        приоритетом. Возвращает None, если такой переменной нет.
        """
        known = set(self.session.facts.keys())
        fired_ids = {d.rule.id for d in self.session.decisions}
        best: tuple[int, str] | None = None

        for rule in sorted(self.kb.rules, key=lambda r: -r.priority):
            if rule.id in fired_ids:
                continue
            # Все уже известные условия правила должны выполняться,
            # иначе правило заведомо не сработает и спрашивать бессмысленно.
            ok = True
            for var, op, exp in rule.conditions:
                if var in known:
                    actual = self.session.facts[var]
                    if op == "=" and actual != exp:
                        ok = False
                        break
                    if op == "!=" and actual == exp:
                        ok = False
                        break
                    if op == "in" and actual not in exp:
                        ok = False
                        break
            if not ok:
                continue

            missing = [v for v, _, _ in rule.conditions if v not in known]
            for var in missing:
                if best is None or rule.priority > best[0]:
                    best = (rule.priority, var)

        return best[1] if best else None

    def _print_final(self, fired: FiredRule) -> None:
        from explanation import render_explanation
        print("\n" + "=" * 62)
        print("  РЕЗУЛЬТАТ")
        print("=" * 62)
        print(f"\n{fired.rule.conclusion_text}")
        print(f"\nИсточник: {fired.rule.source}")
        print(f"Правило: {fired.rule.id} (приоритет {fired.rule.priority})")
        print("\n" + render_explanation(fired, self.session))
