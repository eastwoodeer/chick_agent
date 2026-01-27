from chick_agent.agent import SimpleAgent
from chick_agent.core import ChickAgentLLM
from chick_agent.tools import MCPTool

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory


def repr():
    import httpx

    agent = SimpleAgent(
        name="AI助手",
        llm=ChickAgentLLM(client=httpx.Client(trust_env=False)),
        system_prompt="你是一名有用的AI助手",
    )
    session = PromptSession(history=FileHistory("/tmp/.chat.history"))

    while True:
        try:
            user_input = session.prompt("🙈: ").strip()

            if user_input.lower() in ["exit", "quit", "q", "x"]:
                print("退出")
                break
            if not user_input:
                continue
            print(f"{agent.name}: ", end="", flush=True)
            agent.run(user_input, stream=True)
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            break


if __name__ == "__main__":
    repr()
