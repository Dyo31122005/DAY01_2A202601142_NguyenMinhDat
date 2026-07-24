import time
from typing import Callable, Any


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:

    for attempt in range(max_retries + 1):

        try:
            return fn()

        except Exception:

            if attempt == max_retries:
                raise

            time.sleep(
                base_delay * (2 ** attempt)
            )


# --------------------------
# Test
# --------------------------

count = 0


def unstable_function():

    global count

    count += 1

    print("Attempt:", count)


    if count < 3:
        raise ValueError("Temporary error")


    return "Success"



result = retry_with_backoff(
    unstable_function,
    max_retries=3
)


print(result)