import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Load .env
# ============================================================

load_dotenv(Path(__file__).parent / ".env")


OPENAI_MODEL = os.getenv(
    "LAB_MODEL",
    "gpt-4o"
)


# ============================================================
# Task 3.1
# Streaming chatbot with history
# ============================================================

def streaming_chatbot() -> None:


    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


    # lưu lịch sử chat
    history = []


    while True:

        user_msg = input("You: ").strip()


        # thoát chatbot
        if user_msg.lower() in {
            "quit",
            "exit",
            "bye"
        }:
            break


        # thêm câu hỏi user
        history.append(
            {
                "role": "user",
                "content": user_msg
            }
        )


        # gọi API dạng streaming
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
        )


        reply_parts = []


        print("Assistant: ", end="")


        # nhận từng chunk
        for chunk in stream:

            delta = (
                chunk
                .choices[0]
                .delta
                .content
                or ""
            )


            # in ngay khi nhận được
            print(
                delta,
                end="",
                flush=True
            )


            # lưu lại
            reply_parts.append(delta)


        print()


        # ghép toàn bộ câu trả lời
        reply = "".join(reply_parts)


        # lưu assistant response
        history.append(
            {
                "role": "assistant",
                "content": reply
            }
        )


        # giữ 3 lượt chat gần nhất
        # mỗi lượt gồm user + assistant
        history = history[-6:]



# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    streaming_chatbot()