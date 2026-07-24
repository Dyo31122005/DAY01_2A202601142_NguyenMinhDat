def compare_models(prompt):

    return {
        "model1": "answer 1",
        "model2": "answer 2"
    }



def batch_compare(prompts):

    results = []

    for prompt in prompts:

        comparison = compare_models(prompt)

        results.append({
            **comparison,
            "prompt": prompt
        })

    return results



if __name__ == "__main__":

    prompts = [
        "Explain AI",
        "Explain ML",
        "Explain NLP"
    ]


    results = batch_compare(prompts)


    for r in results:
        print(r)