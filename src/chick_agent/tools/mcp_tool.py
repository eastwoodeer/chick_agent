import asyncio

from concurrent import futures
from typing import override

from chick_agent.tools.tool import Tool, ToolParameter
from chick_agent.protocols.mcp import MCPClient


class MCPTool(Tool):
    def __init__(
        self,
        name: str = "mcp",
        description: str | None = None,
        server_command: list[str] | None = None,
        server_args: list[str] | None = None,
        server: object | None = None,
        auto_expand: bool = True,
        env: dict[str, str] | None = None,
    ):
        self.name = name
        self.server_command = server_command
        self.server_args = server_args or []
        self.server = server
        self.auto_expand = auto_expand
        self.prefix = f"{name}_" if auto_expand else ""
        self._client = None
        self._available_tools = []
        self.env = env
        super().__init__(name=name, description=description)

    def auto_expand_tools(self) -> list[Tool] | None:
        if not self.auto_expand:
            return None
        self._discover_tools()
        tools = []
        for mcp_tool in self._available_tools:
            tool = MCPInnerTool(
                self,
                mcp_tool,
            )
            tools.append(tool)
        return tools

    def _discover_tools(self):
        try:

            async def discover():
                source = self.server if self.server else self.server_command
                async with MCPClient(
                    source, server_args=self.server_args, env=self.env
                ) as client:
                    tools = await client.list_tools()
                    return tools

            try:
                _ = asyncio.get_running_loop()

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(discover())
                    finally:
                        new_loop.close()

                with futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    self._available_tools = future.result()
            except RuntimeError:
                self._available_tools = asyncio.run(discover())

        except Exception as e:
            self._available_tools = []

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: list_tools, call_tool, list_resources, read_resource, list_prompts, get_prompt",
                required=True,
            ),
            ToolParameter(
                name="tool_name",
                type="string",
                description="工具名称（call_tool 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="arguments",
                type="object",
                description="工具参数（call_tool 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="uri",
                type="string",
                description="资源 URI（read_resource 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="prompt_name",
                type="string",
                description="提示词名称（get_prompt 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="prompt_arguments",
                type="object",
                description="提示词参数（get_prompt 操作可选）",
                required=False,
            ),
        ]

    @override
    def run(self, parameters: dict[str, object]) -> str:
        print("dododo")
        action = parameters.get("action", "").lower()

        if not action:
            return "错误：必须指定 action 参数或 tool_name 参数"
        try:

            async def run_mcp_tool():
                print("run_mcp_tool")
                if self.server:
                    client_source = self.server
                else:
                    client_source = self.server_command

                print(client_source)
                async with MCPClient(
                    client_source, self.server_args, env=self.env
                ) as client:
                    if action == "list_tools":
                        tools = await client.list_tools()
                        if not tools:
                            return "没有找到可用工具"
                        result = f"找到 {len(tools)} 个工具:\n"
                        for tool in tools:
                            result += (
                                f"- {tool.get('name')}: {tool.get('description')}\n"
                            )
                        return result
                    elif action == "call_tool":
                        tool_name: str = parameters.get("tool_name")
                        if not tool_name:
                            return "错误: 没有指定tool_name"
                        arguments = parameters.get("arguments", {})
                        result = await client.call_tool(tool_name, arguments)
                        return f"工具 {tool_name} 执行结果: \n{result}"
                    else:
                        return "错误: 不支持的操作: {action}"

            try:
                _ = asyncio.get_running_loop()

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(run_mcp_tool())
                    finally:
                        new_loop.close()

                with futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            except RuntimeError:
                print("runtimeerror....")
                return asyncio.run(run_mcp_tool())

        except Exception as e:
            return f"MCP操作失败: {e}"


class MCPInnerTool(Tool):
    def __init__(self, mcp_tool: Tool, tool_info: dict[str, object]):
        self.mcp_tool = mcp_tool
        self.tool_info = tool_info
        self.mcp_tool_name = tool_info.get("name", "unknown")
        self._parameters = self._parse_input_schema(tool_info.get("input_schema", {}))

        super().__init__(
            name=f"{self.mcp_tool_name}",
            description=tool_info.get("description", f"MCP工具: {self.mcp_tool_name}"),
        )

    def _parse_input_schema(
        self, input_schema: dict[str, object]
    ) -> list[ToolParameter]:
        parameters = []
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_description = param_info.get("description", "")
            is_required = param_name in required

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=param_description,
                    required=is_required,
                )
            )
        return parameters

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return self._parameters

    @override
    def run(self, params: dict[str, object]) -> str:
        mcp_params = {
            "action": "call_tool",
            "tool_name": self.mcp_tool_name,
            "arguments": params,
        }
        return self.mcp_tool.run(mcp_params)
