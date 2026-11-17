import json

import httpx
import pytest

from llm.providers.cloudru import CloudRuEmbedding, CloudRuError


def test_cloudru_contract_and_order():
    def handle(request):
        assert str(request.url) == "https://foundation-models.api.cloud.ru/v1/embeddings"
        assert request.headers["authorization"] == "Bearer private-test-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "ai-sage/Giga-Embeddings-instruct-3B",
            "input": ["a", "b"],
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    dict(index=1, embedding=[0.2] * 2048),
                    dict(index=0, embedding=[0.1] * 2048),
                ]
            },
        )

    adapter = CloudRuEmbedding(
        api_key="private-test-key", transport=httpx.MockTransport(handle)
    )
    vectors = adapter.embed_documents(["a", "b"])
    assert vectors[0][0] == 0.1 and vectors[1][0] == 0.2


@pytest.mark.parametrize(
    "data",
    [
        [],
        [dict(index=0, embedding=[1.0])],
        [dict(index=0, embedding=[0.0] * 2048)],
        [dict(index=1, embedding=[1.0] * 2048)],
        [dict(index=True, embedding=[1.0] * 2048)],
        [dict(index=0, embedding=[True] * 2048)],
    ],
)
def test_invalid_response_fails_closed(data):
    adapter = CloudRuEmbedding(
        api_key="secret",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": data})),
    )
    with pytest.raises(CloudRuError):
        adapter.embed_documents(["a"])


def test_upstream_error_hides_secrets_and_does_not_retry():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429, text="secret upstream body")

    adapter = CloudRuEmbedding(api_key="secret", transport=httpx.MockTransport(handle))
    with pytest.raises(CloudRuError) as caught:
        adapter.embed_documents(["a"])
    assert len(calls) == 1
    assert caught.value.status_code == 429
    assert "secret" not in str(caught.value)


def test_factory_and_cache_configuration_follow_cloudru(monkeypatch):
    from config.instance import install
    from faq.matcher import FAQMatcher
    from llm.factory import get_faq_provider
    from llm.providers.cloudru import CloudRuProvider

    install(
        {
            "LLM_PROVIDER": "deepseek",
            "FAQ_EMBEDDING_PROVIDER": "cloudru",
            "CLOUDRU_API_KEY": "first-key",
            "CLOUDRU_EMBEDDING_MODEL": "ai-sage/Giga-Embeddings-instruct-3B",
        },
        1,
    )
    first = get_faq_provider()
    assert isinstance(first, CloudRuProvider)
    assert FAQMatcher._embedding_configuration()[0] == "cloudru"
    install(
        {
            "LLM_PROVIDER": "deepseek",
            "FAQ_EMBEDDING_PROVIDER": "cloudru",
            "CLOUDRU_API_KEY": "second-key",
            "CLOUDRU_EMBEDDING_MODEL": "ai-sage/Giga-Embeddings-instruct-3B",
        },
        2,
    )
    assert get_faq_provider() is not first


def test_mismatched_response_model_is_rejected():
    adapter = CloudRuEmbedding(
        api_key="secret",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "model": "other",
                    "data": [{"index": 0, "embedding": [0.1] * 2048}],
                },
            )
        ),
    )
    with pytest.raises(CloudRuError, match="другую модель"):
        adapter.embed_query("a")
