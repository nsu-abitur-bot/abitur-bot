from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        config_file = Path(__file__).parent / config_path
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Ошибка парсинга YAML: {e}")


def get_parser_config() -> dict:
    config = load_config()
    return config.get("parser", {})
