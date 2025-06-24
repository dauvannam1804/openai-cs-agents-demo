import asyncio
from agents import Agent, run_demo_loop
from agents import OpenAIChatCompletionsModel

from agents.model_settings import ModelSettings
from openai import AsyncOpenAI


# Define the external client
external_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required, but unused
)


local_model = OpenAIChatCompletionsModel(
    model="llama3.2:3b",
    openai_client=external_client,
)


async def main() -> None:
    agent = Agent(
        name="Assistant", model=local_model, instructions="You are a helpful assistant."
    )
    await run_demo_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
