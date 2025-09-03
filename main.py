import asyncio
from bot.main import main as bot_main


async def main():
    """Запускает bot."""
    # Запускаем bot
    bot_task = asyncio.create_task(bot_main())

    # Ждем завершения задач
    await asyncio.gather(bot_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Программа остановлена")
