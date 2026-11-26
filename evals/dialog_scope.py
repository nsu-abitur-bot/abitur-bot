"""Synthetic corpus and model outputs shared by the #481 regression checks."""

from llm.base import LLMResult

OLD_QUESTION = "Когда начинается приём в магистратуру ФИТ?"
QUESTION = "Когда начинается приём на «Бизнес-информатика» на бюджет в 2026 году?"
DOCUMENTS_QUESTION = "А какие документы нужны?"
PORTAL_QUESTION = "А как подать заявление через личный кабинет?"
FACULTY_QUESTION = "А когда начинается приём в магистратуру ММФ?"
DEADLINE = "до 18 июля"
PAID_DEADLINE = "до 23 июля"
OTHER_YEAR_DEADLINE = "до 21 июля"
DOCUMENT = "согласие на зачисление"
PORTAL = "код подтверждения из письма"
MASTER_DEADLINE = "до 12 августа"
DEADLINE_URL = "https://example.test/dialog/deadline"
PAID_URL = "https://example.test/dialog/paid"
OTHER_YEAR_URL = "https://example.test/dialog/other-year"
DOCUMENT_URL = "https://example.test/dialog/documents"
PORTAL_URL = "https://example.test/dialog/portal"
MASTER_URL = "https://example.test/dialog/master"

_CORPUS = {
    DEADLINE_URL: f"Бизнес-информатика — бакалавриат, 38.03.05, бюджет 2026 {DEADLINE}.",
    PAID_URL: f"Бизнес-информатика — бакалавриат, 38.03.05, платно 2026 {PAID_DEADLINE}.",
    OTHER_YEAR_URL: (
        f"Бизнес-информатика — бакалавриат, 38.03.05, платно 2025 {OTHER_YEAR_DEADLINE}."
    ),
    DOCUMENT_URL: f"Бизнес-информатика — бакалавриат, 38.03.05, нужно {DOCUMENT}.",
    PORTAL_URL: f"Бизнес-информатика — бакалавриат, 38.03.05, введите {PORTAL}.",
    MASTER_URL: f"ММФ — магистратура, приём документов {MASTER_DEADLINE}.",
}
CHUNKS = [{"file_path": url, "content": content} for url, content in _CORPUS.items()]


class DialogProvider:
    async def generate(self, messages, profile=None):
        system, user = (str(message.content) for message in messages)
        if "Переформулируй вопрос" in system:
            return f"{user.rstrip('?')} (правила приёма)"
        request, _, chunk = user.partition("\n\nФрагмент:\n")
        question = request.rsplit("Вопрос: ", 1)[-1].lower()
        wanted = ""
        if "личный кабинет" in question:
            wanted = PORTAL
        elif "ммф" in question:
            wanted = MASTER_DEADLINE
        elif "Бизнес-информатика" in request:
            if "документы" in question:
                wanted = DOCUMENT
            elif "другом году" in question and "платное" in request:
                wanted = OTHER_YEAR_DEADLINE
            elif "платное" in question and "бюджет" not in request:
                wanted = PAID_DEADLINE
            elif "бюджет" in question:
                wanted = DEADLINE
        return '{"score": 1}' if wanted and wanted in chunk else '{"score": 0}'

    async def generate_with_tools(self, messages, **kwargs):
        context = str(messages[0].content)
        for fact in (
            DEADLINE,
            PAID_DEADLINE,
            OTHER_YEAR_DEADLINE,
            DOCUMENT,
            PORTAL,
            MASTER_DEADLINE,
        ):
            if fact in context:
                return LLMResult(text=f"Подтверждённый факт: {fact}.")
        return LLMResult(text="Информация не найдена.")
