import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")

def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 1.2,
    top_p: float = 0.9,
    max_tokens: int = 10000,
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    latency_seconds = time.perf_counter() - start_time
    response_text = response.choices[0].message.content

    return response_text, latency_seconds


if __name__ == "__main__":
    question = "Hãy kể cho tôi một sự thật thú vị về Hà Nội."

    answer, latency = call_openai(question)

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print(f"\nLatency: {latency:.3f} seconds")