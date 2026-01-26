import re

from collections.abc import Iterator
from chick_agent.core.agent import Agent
from chick_agent.core.config import Config
from chick_agent.core.llm import ChickAgentLLM
from chick_agent.core.message import Message
from chick_agent.tools import ToolRegistry, Tool, tool

TOOL_USAGE_PROMPT = """
{basic_prompt}

## 可用工具
你可以使用以下工具来帮助回答问题
{tools_description}

## 工具调用格式
当需要调用工具时, 请使用以下格式:
`[TOOL_CALL:{{tool_name}}:{{parameters}}]`

### 参数格式说明
1. **多个参数**：使用 `key=value` 格式，用逗号分隔
   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`
   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`
2. **单个参数**：直接使用 `key=value`
   示例：`[TOOL_CALL:search:query=Python编程]`

### 重要提示
- 参数名必须与工具定义的参数名完全匹配
- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=12`
- 文件路径等字符串参数直接写：`path=README.md`
- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答
"""


class SimpleAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: ChickAgentLLM,
        system_prompt: str | None = None,
        tool_registry: ToolRegistry | None = None,
        config: Config | None = None,
    ):
        self.enable_tool_calling = False
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        messages = []
        enhanced_prompt = self._get_system_tool_prompt()
        messages.append({"role": "system", "content": enhanced_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            current_iteration += 1
            response = self.llm.invoke(messages, **kwargs)
            tool_calls = self._parse_tool_calls(response)
            if tool_calls:
                tool_results = []
                prev_response = response
                for call in tool_calls:
                    result = self._execute_tool_call(
                        call["tool_name"], call["parameters"]
                    )
                    tool_results.append(result)
                    # 删除此次工具调用
                    prev_response = prev_response.replace(call["original"], "")
                messages.append({"role": "assistant", "content": prev_response})
                tool_results_text = "\n\n".join(tool_results)
                messages.append(
                    {
                        "role": "user",
                        "content": f"工具执行结果: \n{tool_results_text}\n\n请基于这些结果给出完整的答复",
                    }
                )
                continue
            final_response = response
            break
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        return final_response

    def _execute_tool_call(self, tool_name: str, tool_parameters: str) -> str:
        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"错误: 未找到工具 {tool_name}"
            params = self._parse_tool_parameters(tool_name, tool_parameters)
            result = tool.run(params)
            return f"工具 {tool_name} 执行结果\n{result}"
        except Exception as e:
            return f"调用工具 {tool_name} 失败"

    def _parse_tool_parameters(
        self, tool_name: str, parameters: str
    ) -> dict[str, object]:
        params = {}
        if "=" in parameters:
            if "," in parameters:
                parameters_pairs = parameters.split(",")
                for pair in parameters_pairs:
                    n, v = pair.split("=", maxsplit=1)
                    params[n.strip()] = v.strip()
            else:
                n, v = parameters.split("=", maxsplit=1)
                params[n.strip()] = v.strip()
            params = self._convert_parameter_types(tool_name, params)
        return params

    def _convert_parameter_types(
        self, tool_name: str, params: dict[str, object]
    ) -> dict[str, object]:
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return params

        tool_parameters = tool.get_parameters()
        if not tool_parameters:
            return params

        param_types = {}
        for param in tool_parameters:
            param_types[param.name] = param.type

        converted_params = {}
        for k, v in params.items():
            if k in param_types.keys():
                ty = param_types[k]
                try:
                    if ty == "number" or ty == "integer":
                        # 转换为数字
                        if isinstance(v, str):
                            converted_params[k] = float(v) if ty == "number" else int(v)
                        else:
                            converted_params[k] = v
                    elif ty == "boolean":
                        # 转换为布尔值
                        if isinstance(v, str):
                            converted_params[k] = v.lower() in (
                                "true",
                                "1",
                                "yes",
                                "是",
                            )
                        else:
                            converted_params[k] = bool(v)
                    else:
                        converted_params[k] = v
                except (ValueError, TypeError):
                    # 转换失败，保持原值
                    converted_params[k] = v
            else:
                converted_params[k] = v
        return converted_params

    def _parse_tool_calls(self, text: str) -> list[dict[str, str]]:
        pattern = r"\[TOOL_CALL:([^:]+):([^\]]+)\]"
        matches = re.findall(pattern, text)
        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append(
                {
                    "tool_name": tool_name.strip(),
                    "parameters": parameters.strip(),
                    "original": f"[TOOL_CALL:{tool_name}:{parameters}]",
                }
            )
        return tool_calls

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

        full_prompt = TOOL_USAGE_PROMPT.format(
            basic_prompt=basic_prompt, tools_description=tools_description
        )

        return full_prompt
