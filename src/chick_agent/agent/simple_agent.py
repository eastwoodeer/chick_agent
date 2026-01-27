from typing import override
from chick_agent.agent.basic_agent import BasicAgent
from chick_agent.core.config import Config
from chick_agent.core.llm import ChickAgentLLM
from chick_agent.core.message import Message
from chick_agent.tools import ToolRegistry


class SimpleAgent(BasicAgent):
    def __init__(
        self,
        name: str,
        llm: ChickAgentLLM,
        system_prompt: str | None = None,
        tool_registry: ToolRegistry | None = None,
        config: Config | None = None,
    ):
        super().__init__(name, llm, system_prompt, tool_registry, config)

    @override
    def run(
        self,
        input_text: str,
        stream: bool = False,
        max_tool_iterations: int = 3,
        **kwargs,
    ) -> str:
        messages = []
        enhanced_prompt = self._get_system_tool_prompt()
        messages.append({"role": "system", "content": enhanced_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        if not self.enable_tool_calling:
            response = self._execute_llm(messages, stream, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            current_iteration += 1
            response = self._execute_llm(messages, stream, **kwargs)
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
            final_response = self._execute_llm(messages, stream, **kwargs)

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        return final_response
