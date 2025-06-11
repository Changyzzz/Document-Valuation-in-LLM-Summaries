import openai
import json

def evaluate_with_gpt(combined_summaries, original_query, temperature=0.1):
    """Evaluate the combined summaries using GPT and return scores for each summary."""

    evaluation_prompt = f"""
    You are an AI model trained to evaluate summaries. Below, you will find several summaries identified by their labels.

    Your task is to rate each summary on one metric.

    Please make sure you read and understand every single word of these instructions.

    Evaluation Criteria:
    Information Coverage MUST be an integer from 0 to 10 - How well the summary captures and clearly describes one or several key characteristics of the
    product. A high-quality summary should convey the important features, benefits, or drawbacks of the product as highlighted in the reviews. It should
    provide a rich and accurate depiction of key points.

    Pay attention: The most important consideration is how effectively the summary communicates the product's key characteristics. The clearer and
    more richly it conveys these characteristics, the higher the score. If it fails to adequately describe the product's features, it should receive
    a low score.

    Evaluation Steps:
    1. Read all summaries provided and compare them carefully. Ensure the summary clearly and richly describes the key points relevant to the product
      without including irrelevant information.
    2. Identify any important details or characteristics of the product that are missing from the summary.
    3. Rate each of the summary based on how well it covers and conveys the important information from the reviews. The MORE comprehensively the
    summary covers the relevant information, the HIGHER the score it should receive. Pay attention: The primary focus should be on the topic
    {original_query}. If the summary deviates from the topic, it should receive a low score, regardless of the amount of information it contains.
    4. If a summary contains only the sentence "[X] is not related to the query." where X is a number, then give it a score of 0. However,
    if the summary contains other content besides this sentence, just ignore it when scoring.

    Your response should be in JSON format, with an array of objects. Each object should have two properties:
    1. "key": The key of the summary (e.g., "0", "1", "01", etc.)
    2. "score": The score for that summary (an integer from 0 to 10)
    """

    input_text = "\n\n".join([f"Summary[{key}]: {summary['summary']}" for key, summary in combined_summaries.items()])

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": evaluation_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "evaluation_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "evaluations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "score": {
                                        "type": "integer",
                                        "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                    }
                                },
                                "required": ["key", "score"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["evaluations"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = json.loads(response['choices'][0]['message']['content'])
    initial_scores = {item['key']: item['score'] for item in result['evaluations']}

    #Optional: Adjust scores to ensure larger combinations have scores >= their subsets
    #adjusted_scores = initial_scores.copy()
    #for key in sorted(initial_scores.keys(), key=lambda x: len(x)):
    #    for i in range(1, len(key)):
    #        subsets = [''.join(sorted(comb)) for comb in combinations(key, i)]
    #        for subset in subsets:
    #            if subset in adjusted_scores:
    #                adjusted_scores[key] = max(adjusted_scores[key], adjusted_scores[subset])

    #return adjusted_scores
    return initial_scores