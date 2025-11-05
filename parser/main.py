import json
import logging
from pathlib import Path

from parser.config.load_config import get_parser_config
from parser.nsu_parser import parse_nsu_faculty

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    print("Запуск парсера НГУ...")

    config = get_parser_config()
    faculties = config.get("faculties", {})

    if not faculties:
        print("Список факультетов в конфигурации пуст!")
        return

    results = {}

    for fac_code, fac_path in faculties.items():
        print(f"\nОбработка факультета: {fac_code} ({fac_path})")

        data = parse_nsu_faculty(f"{fac_path}/")

        results[fac_code] = data

        print(f"  Заголовок: {data.get('title', 'Не найден')}")
        print(f"  Всего блоков: {data.get('total_blocks', 0)}")

    output_file = Path("parsed_faculties.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nДанные всех факультетов сохранены в {output_file}")
    print("Парсинг завершен")


if __name__ == "__main__":
    main()
