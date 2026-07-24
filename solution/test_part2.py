import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load .env
# Nếu .env nằm ở H:\Lab_day1\.env
load_dotenv(Path(__file__).parent / ".env")


# Model chính
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")


# ============================================================
# Task 2.1 - Chat with system prompt
# ============================================================

def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


    start = time.perf_counter()


    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


    latency = time.perf_counter() - start


    response_text = response.choices[0].message.content


    return response_text, latency



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    system_prompt = """
    You are an elementary school teacher.
    Explain everything in a simple way.
    """

    user_prompt = """
    Explain Machine Learning.
    """


    answer, latency = chat_with_system_prompt(
        system_prompt,
        user_prompt
    )


    print("System:")
    print(system_prompt)

    print("\nUser:")
    print(user_prompt)

    print("\nAnswer:")
    print(answer)

    print(f"\nLatency: {latency:.3f} seconds")