import os
from pathlib import Path

from dotenv import load_dotenv


# Load .env nếu cần dùng OPENAI_MODEL
load_dotenv(Path(__file__).parent / ".env")


# Model mặc định
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")


# ============================================================
# Task 2.2 - Count tokens
# ============================================================

def count_tokens(
    text: str,
    model: str = OPENAI_MODEL
) -> int:
    """
    Count number of tokens using tiktoken.

    If tiktoken cannot load encoding
    (unsupported model / offline),
    use fallback estimation:
        max(1, len(text)//4)
    """

    try:
        import tiktoken

        # Lấy tokenizer tương ứng với model
        enc = tiktoken.encoding_for_model(model)

        # Encode text -> token ids -> count
        return len(enc.encode(text))

    except Exception:
        # Fallback:
        # trung bình 1 token ~ 4 ký tự
        return max(1, len(text) // 4)



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    texts = [
        "Hello world",
        "Machine Learning is a field of Artificial Intelligence.",
        "Xin chào, tôi đang học Data Science."
    ]


    print("Default model:")
    print(OPENAI_MODEL)


    for text in texts:

        tokens = count_tokens(text)

        print("\nText:")
        print(text)

        print("Tokens:")
        print(tokens)


    print("\nTest unknown model:")

    unknown_tokens = count_tokens(
        "This model does not exist",
        "fake-model-123"
    )

    print(
        "Fallback tokens:",
        unknown_tokens
    )