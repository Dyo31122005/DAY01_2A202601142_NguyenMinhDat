import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# Load .env
# ============================================================

load_dotenv(Path(__file__).parent / ".env")


OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")


# ============================================================
# Bảng giá token (USD / 1000 tokens)
# ============================================================

PRICING_PER_1K_TOKENS = {

    "gpt-4o": {
        "input": 0.005,
        "output": 0.015,
    },

    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006,
    },
}


# ============================================================
# Task 2.2 - Count tokens
# ============================================================

def count_tokens(
    text: str,
    model: str = OPENAI_MODEL
) -> int:

    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)

        return len(enc.encode(text))

    except Exception:

        # fallback:
        # 1 token ~ 4 ký tự
        return max(1, len(text) // 4)



# ============================================================
# Task 2.3 - Estimate cost
# ============================================================

def estimate_cost(
    prompt: str,
    response: str,
    model: str = OPENAI_MODEL
) -> dict:

    # Token input
    input_tokens = count_tokens(
        prompt,
        model
    )


    # Token output
    output_tokens = count_tokens(
        response,
        model
    )


    # Lấy giá model
    # Nếu model không có trong bảng giá
    # dùng giá gpt-4o
    pricing = PRICING_PER_1K_TOKENS.get(
        model,
        PRICING_PER_1K_TOKENS["gpt-4o"]
    )


    # Chi phí input
    input_cost = (
        input_tokens / 1000
    ) * pricing["input"]


    # Chi phí output
    output_cost = (
        output_tokens / 1000
    ) * pricing["output"]


    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":


    prompt = """
    Explain Machine Learning in simple words.
    """


    response = """
    Machine Learning is a branch of Artificial Intelligence
    that allows computers to learn patterns from data.
    """


    print("Model:")
    print(OPENAI_MODEL)


    print("\nPrompt:")
    print(prompt)


    print("\nResponse:")
    print(response)


    result = estimate_cost(
        prompt,
        response
    )


    print("\nCost estimation:")

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )


    print("\nTest unknown model:")

    unknown_result = estimate_cost(
        "Hello",
        "World",
        "fake-model-123"
    )


    print(unknown_result)