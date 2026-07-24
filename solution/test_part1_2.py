import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load file .env
# .env nằm cùng thư mục solution/
load_dotenv(Path(__file__).parent / ".env")


# Model chính
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")

# Model mini
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ============================================================
# Task 1: Call model chính
# ============================================================

def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:

    # Tạo OpenAI client
    # Gemini dùng OpenAI-compatible API nên cần base_url
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )


    # Bắt đầu đo latency
    start_time = time.perf_counter()


    # Gửi request
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


    # Tính thời gian phản hồi
    latency_seconds = time.perf_counter() - start_time


    # Lấy text trả về
    response_text = response.choices[0].message.content


    return response_text, latency_seconds



# ============================================================
# Task 2: Call model mini
# ============================================================

def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:

    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    question = "Explain Machine Learning in 2 sentences."


    print("===== Using main model =====")

    answer, latency = call_openai(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print(f"\nLatency: {latency:.3f} seconds")



    print("\n\n===== Using mini model =====")

    answer_mini, latency_mini = call_openai_mini(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer_mini)

    print(f"\nLatency: {latency_mini:.3f} seconds")