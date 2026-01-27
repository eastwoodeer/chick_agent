import os
import httpx

from typing import Literal
from collections.abc import Iterator

from openai import OpenAI

from chick_agent.core.exceptions import ChickAgentException, LLMException

SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "custom",
]


class ChickAgentLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: SUPPORTED_PROVIDERS | None = None,
        http_client: httpx.Client | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **kwargs,
    ):
        # 优先使用传入参数，如果未提供，则从环境变量加载
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        self.provider = (
            (provider or os.getenv("LLM_PROVIDER", "")).lower() if provider else None
        )
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise ChickAgentException("未找到合适的api_key或api地址")
            return
        self._client = self._create_client(http_client)

    def _resolve_credentials(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> tuple[str | None, str | None]:
        if self.provider == "openai":
            resolved_api_key = (
                api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            )
            resolved_base_url = (
                base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            )
        elif self.provider == "deepseek":
            resolved_api_key = (
                api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
            )
            resolved_base_url = (
                base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
            )
        else:
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
        return (resolved_api_key, resolved_base_url)

    def _get_default_model(self) -> str:
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-reasoner"
        else:
            return "deepseek-chat"

    def _create_client(self, http_client: httpx.Client = None) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=http_client,
        )

    def think(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in response:
                if (not chunk.choices) or len(chunk.choices) == 0:
                    break
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            print()
        except Exception as e:
            raise LLMException(f"调用 {self.model} 模型失败: {e}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["temperature", "max_tokens"]
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMException(f"调用 {self.model} 模型失败: {e}")


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一名AI助理"},
        {"role": "user", "content": "给我讲一个故事"},
    ]
    for c in nc.think(messages):
        print(c, end="", flush=True)
