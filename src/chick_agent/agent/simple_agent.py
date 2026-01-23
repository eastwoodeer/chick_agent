from collections.abc import Iterator
from chick_agent.core.agent import Agent
from chick_agent.core.config import Config
from chick_agent.core.llm import ChickAgentLLM
from chick_agent.core.message import Message
from chick_agent.tools import ToolRegistry, Tool, tool


class SimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: ChickAgentLLM,
        system_prompt: str | None = None,
        tool_registry: ToolRegistry | None = None,
        config: Config | None = None,
    ):
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        messages = []
        enhanced_prompt = self._get_system_tool_prompt()
        print(enhanced_prompt)
        messages.append({"role": "system", "content": enhanced_prompt})

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

    def add_tool(self, tool: Tool, auto_expand: bool = True):
        self.enable_tool_calling = True
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def _get_system_tool_prompt(self) -> str:
        basic_prompt = self.system_prompt or "你是一名有用的AI助手"
        tools_description = self.tool_registry.get_tool_descriptions()
        if not tools_description or tools_description == "无可用工具":
            return basic_prompt
        full_prompt = f"{basic_prompt}\n\n## 可用工具\n"
        full_prompt += "你可以使用以下工具来帮助回答问题\n"
        full_prompt += tools_description
        full_prompt += "\n## 工具调用格式\n"
        full_prompt += "当需要调用工具时, 请使用以下格式:\n"
        full_prompt += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"
        full_prompt += "### 参数格式说明\n"
        full_prompt += "1. **多个参数**：使用 `key=value` 格式，用逗号分隔\n"
        full_prompt += "   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        full_prompt += "   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        full_prompt += "2. **单个参数**：直接使用 `key=value`\n"
        full_prompt += "   示例：`[TOOL_CALL:search:query=Python编程]`\n\n"
        full_prompt += "3. **简单查询**：可以直接传入文本\n"
        full_prompt += "   示例：`[TOOL_CALL:search:Python编程]`\n\n"

        full_prompt += "### 重要提示\n"
        full_prompt += "- 参数名必须与工具定义的参数名完全匹配\n"
        full_prompt += '- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a="12"`\n'
        full_prompt += "- 文件路径等字符串参数直接写：`path=README.md`\n"
        full_prompt += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return full_prompt


if __name__ == "__main__":
    import httpx

    # llm = ChickAgentLLM(model="deepseek-chat", provider="deepseek")
    llm = ChickAgentLLM(client=httpx.Client(trust_env=False))
    agent = SimpleAgent("AI助手", llm, "你是一名有用的AI助手，请用中文回答我的问题")
    for chunk in agent.stream_run("你好，介绍一下你自己"):
        print(chunk, end="", flush=True)
    # print(agent.run("你好，介绍一下你自己"))
    print(agent.get_history())
