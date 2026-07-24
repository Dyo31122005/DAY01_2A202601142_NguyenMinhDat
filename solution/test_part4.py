import os
from pathlib import Path
from typing import Callable, Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).parent / ".env")


OPENAI_MODEL = os.getenv(
    "LAB_MODEL",
    "gemini-2.5-flash"
)



# ============================================================
# Retry
# ============================================================

def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
):

    import time

    for attempt in range(max_retries + 1):

        try:
            return fn()

        except Exception:

            if attempt == max_retries:
                raise

            time.sleep(
                base_delay * (2 ** attempt)
            )



# ============================================================
# Count token
# ============================================================

def count_tokens(
    text,
    model=OPENAI_MODEL
):

    try:

        import tiktoken

        enc = tiktoken.encoding_for_model(model)

        return len(enc.encode(text))


    except Exception:

        return max(
            1,
            len(text)//4
        )



# ============================================================
# Cost
# ============================================================

PRICING_PER_1K_TOKENS = {

    "gpt-4o": {
        "input":0.005,
        "output":0.015
    }

}


def estimate_cost(
    prompt,
    response,
    model=OPENAI_MODEL
):

    input_tokens = count_tokens(
        prompt,
        model
    )

    output_tokens = count_tokens(
        response,
        model
    )


    pricing = PRICING_PER_1K_TOKENS.get(
        model,
        PRICING_PER_1K_TOKENS["gpt-4o"]
    )


    input_cost = (
        input_tokens / 1000
    ) * pricing["input"]


    output_cost = (
        output_tokens / 1000
    ) * pricing["output"]


    return {
        "total_cost":
            input_cost + output_cost
    }



# ============================================================
# Part 4
# ============================================================

def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
):

    if get_input is None:
        get_input = input


    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


    history = []

    turns = 0

    tokens_used = 0

    total_cost = 0.0



    while True:


        if (
            max_turns is not None
            and turns >= max_turns
        ):
            break



        user_msg = get_input()


        if user_msg.strip().lower() in {
            "quit",
            "exit",
            "bye"
        }:
            break



        messages = (
            [
                {
                    "role":"system",
                    "content":persona
                }
            ]
            +
            history
            +
            [
                {
                    "role":"user",
                    "content":user_msg
                }
            ]
        )



        stream = retry_with_backoff(
            lambda:
            client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True
            )
        )



        reply_parts = []


        print("Assistant: ", end="")


        for chunk in stream:

            delta = (
                chunk
                .choices[0]
                .delta
                .content
                or ""
            )

            print(
                delta,
                end="",
                flush=True
            )


            reply_parts.append(delta)


        print()



        reply = "".join(reply_parts)



        history.append(
            {
                "role":"user",
                "content":user_msg
            }
        )


        history.append(
            {
                "role":"assistant",
                "content":reply
            }
        )


        history = history[-8:]


        turns += 1


        tokens_used += (
            count_tokens(user_msg)
            +
            count_tokens(reply)
        )


        total_cost += estimate_cost(
            user_msg,
            reply
        )["total_cost"]



    return {
        "turns": turns,
        "tokens_used": tokens_used,
        "total_cost": total_cost,
        "history": history
    }



# ============================================================
# Test
# ============================================================

if __name__ == "__main__":


    result = run_assistant(
        persona="You are a helpful AI teacher.",
        max_turns=2
    )


    print("\n\nStats:")
    print(result)