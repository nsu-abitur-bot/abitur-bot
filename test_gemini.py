import asyncio
import os

import httpx
from dotenv import load_dotenv

# Подключаем наш парсер из проекта
from llm.pdf_parser import parse_pdf_with_llm


async def test_document_pipeline():
    """
    Симулирует пайплайн, который будет происходить в бэкенде:
    1. Получаем список документов (название + ссылка) из админки (POST: /rag/confirm).
    2. Скачиваем каждый файл по ссылке во временную память.
    3. Парсим скачанный PDF через наш LLM-парсер (с сохранением структуры таблиц).
    4. (В проде) Загружаем полученный Markdown в граф RAG.
    """
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Ошибка: GEMINI_API_KEY не найден в .env")
        return

    # 1. Имитация данных, пришедших из админки после POST-запроса
    # Наш тестовый пример, как выглядит запрос:
    mock_request_data = {
        "title": "Памятка студента",
        "url": "https://example.com/page",
        "documents": [
            {
                "title": "Минимальные баллы для поступающих в НГУ в 2026г.pdf",
                "url": "https://www.nsu.ru/upload/iblock/03f/7kue6jk5db3g04a0pdgre7lf7ed4ele9/%D0%9C%D0%B8%D0%BD%D0%B8%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5%20%D0%B1%D0%B0%D0%BB%D0%BB%D1%8B%20%D0%B4%D0%BB%D1%8F%20%D0%BF%D0%BE%D1%81%D1%82%D1%83%D0%BF%D0%B0%D1%8E%D1%89%D0%B8%D1%85%20%D0%B2%202026%20%D0%B3%D0%BE%D0%B4%D1%83.pdf",
            }
        ],
    }

    print("🚀 Начинаем отработку пайплайна...")
    print(f"📦 Исходная страница: {mock_request_data['title']}")

    for doc in mock_request_data["documents"]:
        doc_title = doc["title"]
        doc_url = doc["url"]
        print(f"\n⏳ Обработка документа: {doc_title}")
        print(f"📥 1. Скачиваем файл по ссылке: {doc_url}")

        try:
            # 2. Скачиваем файл
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(doc_url, follow_redirects=True)
                response.raise_for_status()
                pdf_bytes = response.content
            print(f"✅ Файл успешно скачан. Размер: {len(pdf_bytes)} байт.")

            # 3. Парсим скачанный файл с помощью нашего LLM (Gemini)
            print("🧠 2. Отправляем байты в parse_pdf_with_llm() ...")
            # Явно передаем 'gemini', либо можно оставить None (возьмет из .env)
            parsed_markdown = await parse_pdf_with_llm(pdf_bytes, provider="gemini")

            print("\n" + "=" * 40)
            print(f"📄 РЕЗУЛЬТАТ ПАРСИНГА ({doc_title}):")
            print("=" * 40)
            print(parsed_markdown)
            print("=" * 40 + "\n")

            # 4. Имитация загрузки в базу RAG (граф)
            print("💾 3. [SIMULATION] Загрузка текста в RAG...")
            print(
                f"---> add_texts_async(texts=[parsed_markdown], source_ids=['{doc_title}'], file_paths=['{doc_url}'])"
            )
            print(f"✅ Документ {doc_title} успешно 'загружен' в граф!")

        except httpx.HTTPError as e:
            print(f"❌ Ошибка скачивания файла {doc_url}: {e}")
        except Exception as e:
            print(f"❌ Ошибка парсинга/обработки {doc_title}: {e}")


if __name__ == "__main__":
    asyncio.run(test_document_pipeline())
