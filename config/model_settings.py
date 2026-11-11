"""Explicit projection of two administrator choices onto legacy consumers."""

MODEL_VERSION = "_model_settings_version"
GENERATION = "_embedding_generation"
MODEL_FIELDS = {"TEXT_PROVIDER", "TEXT_MODEL", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL"}


def materialize_models(values: dict[str, str]) -> dict[str, str]:
    result = dict(values)
    text = result["TEXT_PROVIDER"]
    embedding = result["EMBEDDING_PROVIDER"]
    if text not in {"openai", "gemini", "deepseek"}:
        raise ValueError("Выберите текстового провайдера")
    if embedding not in {"openai", "gemini", "gigachat"}:
        raise ValueError("Выберите провайдера эмбеддингов")
    if not result["TEXT_MODEL"].strip() or not result["EMBEDDING_MODEL"].strip():
        raise ValueError("Укажите название обеих моделей")
    result["LLM_PROVIDER"] = result["LIGHTRAG_LLM_PROVIDER"] = text
    result[text.upper() + "_MODEL"] = result[text.upper() + "_GRAPH_MODEL"] = result[
        "TEXT_MODEL"
    ]
    result["FAQ_EMBEDDING_PROVIDER"] = result["LIGHTRAG_EMBEDDING_PROVIDER"] = embedding
    result[embedding.upper() + "_EMBEDDING_MODEL"] = result["EMBEDDING_MODEL"]
    if embedding != "gigachat":
        result[embedding.upper() + "_GRAPH_EMBEDDING_MODEL"] = result["EMBEDDING_MODEL"]
    result[MODEL_VERSION] = "2"
    return result
