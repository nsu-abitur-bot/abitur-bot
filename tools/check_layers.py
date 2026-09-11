#!/usr/bin/env python3
"""Проверка зависимостей между пакетами: новых кольцевых импортов быть не должно.

Зачем: пакеты проекта разложены по слоям (см. docs/agents/structure.md),
и импорт «вверх» превращается в цикл. Цикл обходят импортом внутри функции, а
такой импорт не виден ни ruff, ни pyrefly — ошибка вылезает только в рантайме,
на исполнении конкретной ветки.

Шесть циклов у нас уже есть, разбираем их по шагам в задаче #310. Поэтому
проверка работает по baseline: падает на паре, которой в baseline нет. Когда
пара исчезает — тоже падает, с требованием обновить baseline, чтобы починенное
нельзя было вернуть обратно. Когда baseline опустеет, файл удаляем и запрет
становится безусловным.

    python tools/check_layers.py                    проверить
    python tools/check_layers.py --write-baseline   зафиксировать текущее состояние
"""

import argparse
import ast
import collections
import os
import sys

# Пакеты первого уровня, между которыми следим за зависимостями. Всё прочее
# (сторонние библиотеки, tools, alembic) не наше дело.
PACKAGES = {
    "abbrev", "api", "bot", "config", "db", "evals",
    "faq", "llm", "parser", "pipeline", "rag", "scheduler",
}

# Каталоги, в которые не ходим. tests исключены сознательно: тесты имеют право
# импортировать что угодно, цикл в проде от этого не появляется.
SKIP_DIRS = {".git", ".venv", "node_modules", "data", ".ruff_cache", "tests"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "tools", "layers_baseline.txt")


def imported_packages(path: str) -> set[str]:
    """Пакеты из PACKAGES, которые импортирует файл (на любом уровне вложенности)."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Относительные импорты (level > 0) внутри пакета нас не интересуют.
            modules = [node.module] if node.module and node.level == 0 else []
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            top = module.split(".")[0]
            if top in PACKAGES:
                found.add(top)
    return found


def build_graph() -> dict[str, set[str]]:
    """Граф зависимостей между пакетами: пакет -> пакеты, которые он импортирует."""
    graph: dict[str, set[str]] = collections.defaultdict(set)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            relative = os.path.relpath(os.path.join(dirpath, filename), ROOT)
            source = relative.replace(os.sep, "/").split("/")[0]
            if source not in PACKAGES:
                continue
            for target in imported_packages(os.path.join(dirpath, filename)):
                if target != source:
                    graph[source].add(target)
    return graph


def mutual_pairs(graph: dict[str, set[str]]) -> set[str]:
    """Пары пакетов, которые импортируют друг друга, как «a <-> b» (a < b)."""
    pairs = set()
    for source, targets in graph.items():
        for target in targets:
            if source in graph.get(target, ()):
                low, high = sorted((source, target))
                pairs.add(f"{low} <-> {high}")
    return pairs


def read_baseline() -> set[str]:
    if not os.path.exists(BASELINE):
        return set()
    with open(BASELINE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="перезаписать baseline текущим состоянием",
    )
    args = parser.parse_args()

    found = mutual_pairs(build_graph())

    if args.write_baseline:
        with open(BASELINE, "w", encoding="utf-8") as f:
            f.writelines(f"{pair}\n" for pair in sorted(found))
        print(f"baseline записан: {len(found)} пар")
        return 0

    allowed = read_baseline()
    added = sorted(found - allowed)
    removed = sorted(allowed - found)

    print(f"взаимных пар: {len(found)}, разрешено baseline: {len(allowed)}")

    if added:
        print("\nПоявился кольцевой импорт между пакетами:")
        for pair in added:
            print(f"  {pair}")
        print(
            "\nИмпортировать можно только вниз по слоям — см. раздел «Слои и\n"
            "зависимости» в docs/agents/structure.md. Если зависимость нужна,\n"
            "передавай её параметром или переноси код в слой выше.\n"
            "Разбор оставшихся циклов — задача #310."
        )
        return 1

    if removed:
        print("\nПара исчезла — цикл починен:")
        for pair in removed:
            print(f"  {pair}")
        print("\nОбнови baseline: python tools/check_layers.py --write-baseline")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
