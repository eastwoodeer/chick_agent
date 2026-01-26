from chick_agent.agent import SimpleAgent
from chick_agent.core import ChickAgentLLM


def repr():
    import httpx

    agent = SimpleAgent(
        name="AI助手",
        llm=ChickAgentLLM(client=httpx.Client(trust_env=False)),
        system_prompt="你是一名有用的AI助手",
    )

    while True:
        try:
            user_input = input("🙈: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("退出")
                break
            if not user_input:
                continue
            response = agent.run(user_input)
            print(f"{agent.name}: {response}")
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            break


if __name__ == "__main__":
    repr()
