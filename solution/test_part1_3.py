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

# Bảng giá ước tính (USD / 1K token)
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.0001, "output": 0.0004},
}


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
# Task 3: So sánh model chính vs model mini
# ============================================================

def compare_models(prompt: str) -> dict:

    # Gọi lần lượt 2 model với cùng 1 prompt
    gpt4o_answer, gpt4o_time = call_openai(prompt)
    mini_answer, mini_time = call_openai_mini(prompt)

    # Ước tính chi phí output của model chính
    # 0.75 từ ≈ 1 token (ước lượng thô, Task 2.3 sẽ tính chính xác bằng tiktoken)
    # model hiện tại không có trong bảng giá (vd: gemini-flash-lite-latest)
    # thì lấy giá gpt-4o làm tham chiếu
    pricing = PRICING_PER_1K_TOKENS.get(
        OPENAI_MODEL, PRICING_PER_1K_TOKENS["gpt-4o"]
    )
    gpt4o_cost = (len(gpt4o_answer.split()) / 0.75) / 1000 * pricing["output"]

    return {
        "gpt4o_answer": gpt4o_answer,
        "mini_answer": mini_answer,
        "gpt4o_time": gpt4o_time,
        "mini_time": mini_time,
        "gpt4o_cost": gpt4o_cost,
    }



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    question = "Explain Machine Learning in 2 sentences."

    print("===== Comparing main vs mini model =====")

    result = compare_models(question)

    print("\nQuestion:")
    print(question)

    print("\nGPT-4o / main answer:")
    print(result["gpt4o_answer"])
    print(f"Latency: {result['gpt4o_time']:.3f} seconds")
    print(f"Estimated cost: ${result['gpt4o_cost']:.6f}")

    print("\nMini answer:")
    print(result["mini_answer"])
    print(f"Latency: {result['mini_time']:.3f} seconds")
