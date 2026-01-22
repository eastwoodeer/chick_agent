from collections.abc import Iterator
from chick_agent.core.agent import Agent
from chick_agent.core.config import Config
from chick_agent.core.llm import ChickAgentLLM
from chick_agent.core.message import Message


class SimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: ChickAgentLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
    ):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        response = self.llm.invoke(messages, **kwargs)
        return response

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        messages = []
        messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})
        full_response = ""
        for chunk in self.llm.think(messages):
            full_response += chunk
            yield chunk

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))


if __name__ == "__main__":
    import httpx

    # llm = ChickAgentLLM(model="deepseek-chat", provider="deepseek")
    llm = ChickAgentLLM(client=httpx.Client(trust_env=False))
    agent = SimpleAgent("AI助手", llm, "你是一名有用的AI助手，请用中文回答我的问题")
    for chunk in agent.stream_run("你好，介绍一下你自己"):
        print(chunk, end="", flush=True)
    # print(agent.run("你好，介绍一下你自己"))
    print(agent.get_history())
