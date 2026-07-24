def format_comparison_table(results):

    def shorten(text):

        return (
            text
            if len(text) <= 40
            else text[:37] + "..."
        )


    rows = [
        "Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency"
    ]


    for result in results:

        rows.append(
            " | ".join(
                [
                    shorten(result["prompt"]),
                    shorten(result["gpt4o_answer"]),
                    shorten(result["mini_answer"]),
                    f"{result['gpt4o_time']:.2f}s",
                    f"{result['mini_time']:.2f}s",
                ]
            )
        )


    return "\n".join(rows)



if __name__ == "__main__":


    results = [
        {
            "prompt": "Explain Machine Learning",
            "gpt4o_answer":
                "Machine Learning is a branch of Artificial Intelligence that allows computers to learn from data.",

            "mini_answer":
                "Machine Learning helps computers learn patterns.",

            "gpt4o_time": 1.2567,
            "mini_time": 0.8345
        }
    ]


    table = format_comparison_table(results)


    print(table)