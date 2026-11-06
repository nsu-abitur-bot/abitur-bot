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
_SCORE = re.compile(r"(?:проходн\w*|средн\w*)\s+бал", re.I)
_SHORT = re.compile(
    r"^(?:а\s+)?(?:(?:средн\w*|проходн\w*)"
    r"(?=\s*(?:$|[?!.,]|бал|како|на\b|по\b|в\b|за\b))|"
    r"в\s+20\d{2}|за\s+20\d{2}|"
    r"на\s+(?:плат\w*|бюджет\w*|магистр\w*|бакалавр\w*|спо))\b",
    re.I,
)


@dataclass
class ScoreGuard:
    """Фильтры из реплик пользователя; прошлые ответы не являются данными SQL."""

    filters: dict = field(default_factory=dict)
    verified: bool = False
    attempted: bool = False
    clarify: bool = False

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
        short = bool(_SHORT.search(current))
        if not direct and not short:
            return None

        chain = [current]
        previous = questions[:-1] if short else []
        for question in reversed(previous):
            if _SCORE.search(question):
                chain.insert(0, question)
                if not _SHORT.search(question.strip()) or _CODE.search(question):
                    break
                continue
            if _SHORT.search(question.strip()):
                chain.insert(0, question)
            else:
                break
        saved_scope: dict = {}
        scope_updates = [current]
        if short:
            for message in reversed(messages[:-1]):
                metadata = message.additional_kwargs.get("admission_scores")
                if message.type == "ai" and isinstance(metadata, dict):
                    saved_scope = {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in ("program", "faculty", "level", "year", "form", "metric")
                    }
                    break
                if message.type == "human":
                    question = str(message.content).lower().strip()
                    if not _SHORT.search(question):
                        break
                    scope_updates.insert(0, question)
        anchored = bool(saved_scope) or any(_SCORE.search(q) for q in chain[:-1])
        if short and not direct and not anchored:
            # «А в 2024?» после вопроса об общежитии не является запросом баллов.
            if not re.search(r"средн|проходн", current):
                return None
            return cls(clarify=True)

        filters: dict = {"metric": "passing", **saved_scope}
        for question in scope_updates if saved_scope else chain:
            codes = _CODE.findall(question)
            if len(set(codes)) == 1:
                if codes[0] != filters.get("program"):
                    filters.pop("faculty", None)
                filters["program"] = codes[0]
            elif codes:
                filters.pop("program", None)
            years = _YEAR.findall(question)
            if len(set(years)) == 1:
                filters["year"] = int(years[0])
            elif years:
                filters.pop("year", None)
            if "средн" in question:
                filters["metric"] = "average"
            if "проходн" in question:
                filters["metric"] = "both" if "средн" in question else "passing"
            if re.search(r"плат|договор", question):
                filters["form"] = "paid"
            elif "бюджет" in question:
                filters["form"] = "budget"
            for pattern, level in (
                (r"магистр", "master"),
                (r"бакалавр", "bachelor"),
                (r"специалитет", "specialist"),
                (r"\bспо\b", "spo"),
            ):
                if re.search(pattern, question):
                    filters["level"] = level
        return cls(filters=filters)

    def recovery_tool(self, tool: ToolSpec) -> ToolSpec:
        """Одна восстановительная попытка с обязательными известными фильтрами."""
        parameters = deepcopy(tool.parameters)
        for name, value in self.filters.items():
            parameters["properties"][name]["enum"] = [value]
        parameters["required"] = list(self.filters)
        if self.filters.get("program") and not self.filters.get("faculty"):
            parameters["properties"].pop("faculty", None)
        return ToolSpec(tool.name, tool.description, parameters)

    def accepts(self, arguments: dict, *, check_names: bool = True) -> bool:
        """Полная выборка покрывает фильтр, противоположный фильтр — нет."""
        metric = arguments.get("metric", "passing")
        if metric not in (self.filters.get("metric"), "both"):
            return False
        if (
            check_names
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
