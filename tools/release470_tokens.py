"""Offline V4.1 prompt encoding, including roles and function-call framing."""

import gzip
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from tools.admission_usage import IndexingLimit

TOKENIZER_SHA256 = "81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99"
RECIPE_VERSION = "0.2.0"
MODEL = "DeepSeek-V4.1-Flash-1017"


class OfficialCounter:
    """Never downloads files or sends text to a counting service."""

    def __init__(self) -> None:
        try:
            installed = version("deepseek-recipe")
        except PackageNotFoundError:
            raise IndexingLimit("The reviewed offline encoder is not installed") from None
        if installed != RECIPE_VERSION:
            raise IndexingLimit("The reviewed prompt encoder version is required")
        recipe = import_module("deepseek_recipe")

        path = Path(__file__).parent / "release470_resources/tokenizer-v41.json.gz"
        raw = gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
            raise IndexingLimit("Public tokenizer integrity check failed")
        self.encoder = recipe.DeepseekV41Encoding().with_tokenizer(
            recipe.Tokenizer.from_str(raw.decode())
        )

    @property
    def evidence(self) -> dict:
        return {
            "kind": "official-v41-encoding",
            "recipe": RECIPE_VERSION,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "model": MODEL,
        }

    def __call__(self, payload: dict) -> int:
        recipe = import_module("deepseek_recipe")

        fields = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "thinking",
            "stream",
            "stream_options",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
        }
        if set(payload) - fields or payload.get("model") != "deepseek-flash":
            raise IndexingLimit("Unsupported prompt contract")
        if payload.get("thinking") != {"type": "disabled"}:
            raise IndexingLimit("Only the non-thinking text contract is reviewed")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise IndexingLimit("A nonempty text conversation is required")
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message)
                - {"role", "content", "tool_calls", "tool_call_id", "name"}
                or message.get("role") not in {"user", "assistant", "system", "tool"}
                or not isinstance(message.get("content", ""), (str, type(None)))
            ):
                raise IndexingLimit("Multimodal, prefix and unknown framing are refused")
        tools = payload.get("tools", [])
        if not isinstance(tools, list) or any(t.get("type") != "function" for t in tools):
            raise IndexingLimit("Hosted tools are outside the reviewed contract")
        # The API alias is pinned to its documented V4.1 encoding, not V3/tiktoken.
        data = {**payload, "model": "DeepSeek-V4.1-Flash"}
        try:
            converted = recipe.ChatCompletionRequest(data).convert(
                recipe.ConversionOptions()
            )
            count = len(self.encoder.encode(converted.conversation))
        except Exception:
            raise IndexingLimit("Unsupported V4.1 request encoding") from None
        if count < 1:
            raise IndexingLimit("Empty or invalid prompt encoding")
        return count
