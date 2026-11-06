"""Проверка источника баллов до публикации ответа модели."""

import re
from copy import deepcopy
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from llm.base import ToolSpec
from pipeline.tools.admission_scores import AdmissionScoresResult

SCORES_TOOL = "get_admission_scores"
SCORES_UNAVAILABLE = "Не удалось проверить баллы. Попробуйте повторить запрос позже."
SCORES_CLARIFY = "Уточните направление подготовки, для которого нужны баллы."
_CODE = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\b")
_YEAR = re.compile(r"\b20\d{2}\b")
_REFERENCE_SUBJECT = re.compile(
    r"^(?:эт(?:от|а|о|и|ого|ой|их|ому|им|у|ими|ом)|"
    r"т(?:от|а|о|е|ого|ой|ех|ому|ем|у|еми|ом)|"
    r"н(?:ему|его|ём|ем|ей|её|ее|им|их))\b",
    re.I,
)
_ALL_PROGRAMS = re.compile(
    r"\bвс(?:е|ех|ем|еми)\s+(?:образовательн\w*\s+)?"
    r"(?:программ|профил|направлен)\w*\b",
    re.I,
)
_SUBJECT_FILLERS = frozenset(
    "код кода коду кодом коде направление направления направлению направлением "
    "направлении программа программы программе программу программой профиль профиля "
    "профилю профилем профиле специальность специальности подготовки в за на по для "
    "год года году годы годах и или какой какая какие каком там же".split()
)
_SUBJECT_MODIFIER = re.compile(
    r"(?:бюджет(?:а|у|е|ом)?|платн[а-яё]*|платк[а-яё]*|"
    r"бакалавриат[а-яё]*|магистратур[а-яё]*|специалитет[а-яё]*|спо)",
    re.I,
)


def _subject_words(text: str) -> list[str]:
    text = _YEAR.sub("", _CODE.sub("", text.lower()))
    return [
        word
        for word in re.findall(r"[a-zа-яё]+", text)
        if word not in _SUBJECT_FILLERS
        and not _SUBJECT_MODIFIER.fullmatch(word)
        and not _REFERENCE_SUBJECT.fullmatch(word)
    ]


_SCORE = re.compile(r"(?:проходн\w*|средн\w*)\s+бал", re.I)
_SHORT = re.compile(
    r"^(?:а\s+)?(?:(?:средн\w*|проходн\w*)"
    r"(?=\s*(?:$|[?!.,]|бал|како|на\b|по\b|в\b|за\b))|"
    r"в\s+20\d{2}|за\s+20\d{2}|"
    r"(?:на|по)\s+(?:плат\w*|бюджет\w*|магистр\w*|бакалавр\w*|спо|"
    r"\d{2}\.\d{2}\.\d{2}))\b",
    re.I,
)
_GENERIC_SCORE = re.compile(
    r"(?:(?:а|ну|тогда|какой|какие|каков|каковы)\s+)*"
    r"(?:средн\w*|проходн\w*)(?:\s+бал\w*)?"
    r"(?:\s+(?:какой|какие|там|же))?"
    r"(?:\s+(?:в|за)\s+20\d{2}(?:\s+году)?)?[?!.]*",
    re.I,
)


def _is_followup(question: str) -> bool:
    question = re.sub(
        r"^(?:(?:а|какой|какие|каков|каковы)\s+)+", "", question.strip(), flags=re.I
    )
    return bool(
        _SHORT.search(question.strip()) or _GENERIC_SCORE.fullmatch(question.strip())
    )


@dataclass
class ScoreGuard:
    """Фильтры из реплик пользователя; прошлые ответы не являются данными SQL."""

    filters: dict = field(default_factory=dict)
    verified: bool = False
    attempted: bool = False
    clarify: bool = False
    comparisons: dict[str, list] = field(default_factory=dict)
    program_ids: frozenset[str] = frozenset()
    program_names: dict[str, list[str]] = field(default_factory=dict)
    named_program: bool = False

    @classmethod
    def from_messages(cls, messages: list[BaseMessage]) -> "ScoreGuard | None":
        questions = [str(m.content).lower() for m in messages if m.type == "human"]
        if not questions:
            return None
        current = questions[-1].strip(' «"!?.,')
        # Средний балл аттестата и индивидуальные результаты — другой источник.
        if re.search(r"аттестат|диплом|мой средн", current) and "проходн" not in current:
            return None
        direct = bool(_SCORE.search(current))
        short = _is_followup(current)
        if not direct and not short:
            return None

        chain = [current]
        previous = questions[:-1] if short else []
        for question in reversed(previous):
            if _SCORE.search(question):
                chain.insert(0, question)
                if not _is_followup(question) or _CODE.search(question):
                    break
                continue
            if _is_followup(question):
                chain.insert(0, question)
            else:
                break
        saved_scope: dict = {}
        saved_comparisons: dict[str, list] = {}
        program_ids: frozenset[str] = frozenset()
        program_names: dict[str, list[str]] = {}
        scope_updates = [current]
        if short:
            for message in reversed(messages[:-1]):
                metadata = message.additional_kwargs.get("admission_scores")
                if message.type == "ai" and isinstance(metadata, dict):
                    program_ids = frozenset(metadata.get("_program_ids", []))
                    program_names = {
                        key: list(names)
                        for key, names in metadata.get("_program_names", {}).items()
                        if isinstance(names, list)
                    }
                    raw = metadata.get("_comparisons", {})
                    if isinstance(raw, dict):
                        saved_comparisons = {
                            key: list(values)
                            for key, values in raw.items()
                            if key in ("program", "year", "form")
                            and isinstance(values, list)
                        }
                    saved_scope = {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in ("program", "faculty", "level", "year", "form", "metric")
                    }
                    break
                if message.type == "human":
                    question = str(message.content).lower().strip()
                    if not _is_followup(question):
                        break
                    scope_updates.insert(0, question)
        anchored = bool(saved_scope) or any(
            _SCORE.search(q) and not _GENERIC_SCORE.fullmatch(q.strip())
            for q in chain[:-1]
        )
        if not anchored and _GENERIC_SCORE.fullmatch(current):
            return cls(clarify=True)
        if short and not direct and not anchored:
            # «А в 2024?» после вопроса об общежитии не является запросом баллов.
            if not re.search(r"средн|проходн", current):
                return None
            return cls(clarify=True)

        filters: dict = {"metric": "passing", **saved_scope}
        comparisons = saved_comparisons
        needs_program = False
        named_program = False
        for key in comparisons:
            filters.pop(key, None)
        for question in scope_updates if saved_scope else chain:
            subject = re.search(r"\b(?:на|по|для)\s+(.+)", question)
            named_subject = False
            matched_ids: frozenset[str] = frozenset()
            if subject:
                value = subject.group(1)
                words = _subject_words(value)
                named_subject = bool(words)
                if named_subject:
                    named_program = True
                matched_ids = (
                    frozenset(
                        key
                        for key in program_ids
                        if any(
                            words == _subject_words(variant)
                            for name in program_names.get(key, [])
                            for variant in (name, name.rsplit(". ", 1)[-1])
                        )
                    )
                    if words
                    else frozenset()
                )
                if matched_ids:
                    program_ids = matched_ids
                    needs_program = False
                elif (
                    not _CODE.search(value)
                    and not _REFERENCE_SUBJECT.match(value)
                    and not re.match(
                        r"плат|бюджет|бесплат|договор|магистр|бакалавр|спо\b|"
                        r"специалитет|20\d{2}\b",
                        value,
                    )
                ):
                    # Новое название разрешает SQL-инструмент; прежний код
                    # нельзя закреплять в восстановительной схеме вместо него.
                    if filters.get("program") or program_ids:
                        filters.pop("level", None)
                    filters.pop("program", None)
                    filters.pop("faculty", None)
                    comparisons.pop("program", None)
                    program_ids = frozenset()
                    needs_program = False
            codes = _CODE.findall(question)
            if len(set(codes)) == 1:
                if (
                    codes[0] != filters.get("program")
                    or _ALL_PROGRAMS.search(question)
                    or (named_subject and not matched_ids)
                ):
                    filters.pop("faculty", None)
                    if filters.get("program") or program_ids:
                        filters.pop("level", None)
                    program_ids = frozenset()
                filters["program"] = codes[0]
                comparisons.pop("program", None)
                needs_program = False
            elif codes:
                filters.pop("program", None)
                filters.pop("faculty", None)
                filters.pop("level", None)
                comparisons["program"] = sorted(set(codes))
                program_ids = frozenset()
                needs_program = False
            years = _YEAR.findall(question)
            if len(set(years)) == 1:
                filters["year"] = int(years[0])
                comparisons.pop("year", None)
            elif years:
                filters.pop("year", None)
                comparisons["year"] = sorted({int(year) for year in years})
            if "средн" in question:
                filters["metric"] = "average"
            if "проходн" in question:
                filters["metric"] = "both" if "средн" in question else "passing"
            budget = "бюджет" in question or "бесплат" in question
            paid = bool(re.search(r"\bплат|договор", question))
            if budget and paid:
                filters.pop("form", None)
                comparisons["form"] = ["budget", "paid"]
            elif budget:
                filters["form"] = "budget"
                comparisons.pop("form", None)
            elif paid:
                filters["form"] = "paid"
                comparisons.pop("form", None)
            for pattern, level in (
                (r"магистр", "master"),
                (r"бакалавр", "bachelor"),
                (r"специалитет", "specialist"),
                (r"\bспо\b", "spo"),
            ):
                if re.search(pattern, question):
                    if (
                        filters.get("level") != level
                        and (filters.get("program") or program_ids)
                        and not codes
                    ):
                        filters.pop("program", None)
                        comparisons.pop("program", None)
                        program_ids = frozenset()
                        needs_program = not named_subject
                    filters["level"] = level
        return cls(
            filters=filters,
            comparisons=comparisons,
            clarify=needs_program,
            program_ids=program_ids,
            program_names=program_names,
            named_program=named_program,
        )

    def history_scope(self, resolved_scope: dict) -> dict:
        """Сравнение сохраняется как запрос, даже если SQL нашёл один вариант."""
        scope = dict(resolved_scope)
        for key in ("year", "form", "level", "metric"):
            if key in self.filters:
                scope[key] = self.filters[key]
        if self.program_ids:
            scope["_program_ids"] = sorted(self.program_ids)
        ids = scope.get("_program_ids", [])
        names = scope.get("_program_names", {})
        if ids:
            scope["_program_names"] = {
                key: sorted(
                    set(self.program_names.get(key, [])) | set(names.get(key, []))
                )
                for key in ids
                if self.program_names.get(key) or names.get(key)
            }
        if self.comparisons:
            scope["_comparisons"] = deepcopy(self.comparisons)
            for key in self.comparisons:
                scope.pop(key, None)
        return scope

    def recovery_tool(self, tool: ToolSpec) -> ToolSpec:
        """Одна восстановительная попытка с обязательными известными фильтрами."""
        parameters = deepcopy(tool.parameters)
        for name in self.comparisons:
            parameters["properties"].pop(name, None)
        for name, value in self.filters.items():
            parameters["properties"][name]["enum"] = [value]
        parameters["required"] = list(self.filters)
        if self.program_ids:
            ids = sorted(self.program_ids)
            parameters["properties"]["program_ids"] = {
                "type": "array",
                "items": {"type": "string", "enum": ids},
                "minItems": len(ids),
                "maxItems": len(ids),
                "description": "Передай все перечисленные ID выбранных программ.",
            }
            parameters["required"].append("program_ids")
        if self.filters.get("program") and not self.filters.get("faculty"):
            parameters["properties"].pop("faculty", None)
        return ToolSpec(tool.name, tool.description, parameters)

    def accepts(self, arguments: dict, *, check_names: bool = True) -> bool:
        """Полная выборка покрывает фильтр, противоположный фильтр — нет."""
        metric = arguments.get("metric", "passing")
        ids = arguments.get("program_ids")
        if ids is not None and (
            not self.program_ids
            or not isinstance(ids, list)
            or any(not isinstance(value, str) for value in ids)
            or not self.program_ids.issubset(ids)
        ):
            return False
        if any(arguments.get(key) not in (None, "") for key in self.comparisons):
            return False
        if metric not in (self.filters.get("metric"), "both"):
            return False
        if (
            check_names
            and not self.named_program
            and self.filters.get("program")
            and not self.filters.get("faculty")
            and arguments.get("faculty")
        ):
            # Дополнительный факультет способен сузить выборку до чужого
            # факультета, даже когда код направления указан верно.
            return False
        for key in ("program", "faculty", "year", "form", "level"):
            if key in ("program", "faculty") and not check_names:
                continue
            expected = self.filters.get(key)
            actual = arguments.get(key)
            if expected is not None and actual not in (None, ""):
                if key == "program":
                    if str(expected) not in str(actual):
                        return False
                elif str(actual) != str(expected):
                    return False
        return True

    def record(self, result: str) -> None:
        self.attempted = True
        self.verified = isinstance(result, AdmissionScoresResult) and result.status in (
            "ok",
            "empty",
        )
