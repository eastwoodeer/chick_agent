from fastmcp import Client, FastMCP
from fastmcp.client.transports import PythonStdioTransport, StdioTransport


class MCPClient:
    def __init__(
        self,
        server_source: str,
        server_args: list[str] | None = None,
        transport_type: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs,
    ):
        self.server_args = server_args or []
        self.transport_type = transport_type
        self.env = env or {}
        self.kwargs = kwargs
        self.client: Client = None
        self.server_source = self._prepare_server_source(server_source)
        self._context_manager = None

    def _prepare_server_source(self, server_source: str | FastMCP):
        if isinstance(server_source, str):
            if server_source.endswith(".py"):
                print(f"使用 PythonStdio 传输: {server_source}")
                return PythonStdioTransport(
                    script_path=server_source,
                    args=self.server_args,
                    env=self.env if self.env else None,
                    **self.kwargs,
                )
            else:
                print(f"使用 Stdio 传输: {server_source}")
                return StdioTransport(
                    command=server_source,
                    args=self.server_args,
                    env=self.env if self.env else None,
                    **self.kwargs,
                )

        return server_source

    async def __aenter__(self):
        print("连接到MCP服务器")
        self.client = Client(self.server_source)
        self._context_manager = self.client
        await self._context_manager.__aenter__()
        print("连接成功")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context_manager:
            await self._context_manager.__aexit__(exc_type, exc_val, exc_tb)
            self.client = None
            self._context_manager = None
        print("连接断开")

    async def list_tools(self) -> list[dict[str, object]]:
        if not self.client:
            raise RuntimeError(
                "Client not connected. Use 'async with client:' context manager."
            )

        result = await self.client.list_tools()

        # 处理不同的返回格式
        if hasattr(result, "tools"):
            tools = result.tools
        elif isinstance(result, list):
            tools = result
        else:
            tools = []

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema
                if hasattr(tool, "inputSchema")
                else {},
            }
            for tool in tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> object:
        if not self.client:
            raise RuntimeError(
                "Client not connected. Use 'async with client:' context manager."
            )

        result = await self.client.call_tool(tool_name, arguments)

        # 解析结果 - FastMCP 返回 ToolResult 对象
        if hasattr(result, "content") and result.content:
            if len(result.content) == 1:
                content = result.content[0]
                if hasattr(content, "text"):
                    return content.text
                elif hasattr(content, "data"):
                    return content.data
            return [
                getattr(c, "text", getattr(c, "data", str(c))) for c in result.content
            ]
        return None
