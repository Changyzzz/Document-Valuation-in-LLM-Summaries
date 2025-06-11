import openai
import math
import numpy as np
from itertools import combinations
import json

def label_comment_relevance(filtered_comments, original_query, temperature=0.1):
    """Label each comment based on its relevance to the original query using GPT-4o."""

    relevance_prompt = f"""
Task: You need to determine if each of the following comments is relevant to the topic '{original_query}'. We are working on the amazon gift card.

Instructions:
- For each comment, state whether it is relevant or not by using the format: "[X] is relevant to the query." or "[X] is not related to the query."
- Replace '[X]' with the corresponding comment number.

Comments:
"""

    comments_text = "\n\n".join([f"[{i}] {comment}" for i, comment in enumerate(filtered_comments['combined_text'].tolist())])

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": relevance_prompt},
            {"role": "user", "content": comments_text}
        ],
        temperature=temperature
    )

    relevance_results = response['choices'][0]['message']['content']

    labeled_comments = []
    for i, comment in enumerate(filtered_comments['combined_text'].tolist()):
        if f"[{i}] is relevant to the query." in relevance_results:
            labeled_comments.append({"comment": comment, "label": "relevant"})
        else:
            labeled_comments.append({"comment": comment, "label": "not relevant"})

    return labeled_comments

def generate_final_output(labeled_comments, original_query, temperature=0.1):
    """Generate final output using GPT model with structured API response, using pre-labeled comment relevance."""

    relevant_comments = [item["comment"] for item in labeled_comments if item["label"] == "relevant"]
    not_relevant_indices = [str(i) for i, item in enumerate(labeled_comments) if item["label"] == "not relevant"]
    all_indices = ''.join([str(i) for i in range(len(labeled_comments))])


    if not relevant_comments:
        summary = " ".join([f"[{idx}] is not related to the query." for idx in not_relevant_indices])
        return {"key": all_indices, "summary": summary}

    custom_prompt = f"""
You are tasked with generating a high-quality summary based on user comments. Follow these steps to ensure that your summary is accurate, relevant, and well-structured.

1. Carefully Analyze the Comments:
   - Read through all the comments provided in the context.
   - Identify the key points that are related to the topic '{original_query}'.

2. Select Relevant Information:
   - Only include information in your summary that is relevant to the topic '{original_query}'.
   - For comments marked as "not relevant", simply state "[X] is not related to the query." Replace '[X]' with the corresponding comment number.

3. Construct a Coherent Summary:
   - Use an unbiased and journalistic tone in your summary.
   - Ensure that the summary is medium to long in length and that it covers the key points effectively.

4. Cite the Source of Information:
   - For each part of the summary, include a citation in the form '[NUMBER]', where 'NUMBER' corresponds to the comment's index.
   - Start numbering from '0' and continue sequentially, making sure not to skip any numbers.
   - The citation should be placed at the end of the sentence or clause that it supports.
   - If a sentence in your summary is derived from multiple comments, cite each relevant comment, e.g., '[0][1]'.

5. Final Review:
   - Double-check your citations to ensure they accurately correspond to the comments used.
   - Make sure that every sentence in the summary is cited and that irrelevant comments are correctly identified and excluded after the initial irrelevant statement.
   - Make sure every comment is cited. For example, if comment [0], [1], and [2] are all not related to the topic, then just
   summarize: '[0] is not related to the query. [1] is not related to the query. [2] is not related to the query.'
   If comment [0] is relevant, while [1], [2], and [3] are irrelevant, then summarize like this: provide a summary of [0], and then state '[1] is not related to the query. [2] is not related to the query. [3] is not related to the query.'
   Do not miss any comment even though they are irrelevant.
   - Ensure that your response is structured in JSON format with the following fields:
     - "key": A string that represents the indices of the comments used to generate this summary, e.g., "012" for comments 0, 1, and 2.
     - "summary": The final generated summary text, with citations included.

6. Key Reminders:
   - Do not include any irrelevant information in your summary. If a comment is not related to the topic, state it as described and move on.
   - Ensure that your summary is comprehensive, accurate, and clearly tied to the topic '{original_query}'.
"""

    combined_text = "\n\n".join(relevant_comments)

    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": custom_prompt},
            {"role": "user", "content": combined_text}
        ],
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "summary_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "summary": {"type": "string"}
                    },
                    "required": ["key", "summary"],
                    "additionalProperties": False
                }
            }
        }
    )

    # Build the final summary with irrelevant comments marked
    final_summary = json.loads(response['choices'][0]['message']['content'])
    if not_relevant_indices:
        final_summary["summary"] += " " + " ".join([f"[{idx}] is not related to the query." for idx in not_relevant_indices])

    final_summary["key"] = all_indices

    return final_summary

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

def calculate_shapley_values(filtered_comments, original_query, generate_final_output, evaluate_with_gpt, num_evaluations=4):
    n = len(filtered_comments)
    shapley_values = [0] * n
    cache = {}
    all_summaries = {}

    for size in range(1, n+1):
        for subset in combinations(range(n), size):
            subset_key = ''.join(map(str, subset))
            subset_comments = filtered_comments.iloc[list(subset)]

            # Ensure comments are labeled before generating final output
            labeled_subset_comments = label_comment_relevance(subset_comments, original_query)

            subset_output = generate_final_output(labeled_subset_comments, original_query)
            all_summaries[subset_key] = subset_output
            print(f"summary_{subset_key}_output:")
            print(subset_output['summary'])
            print()

    # Evaluate all summaries multiple times and calculate average scores
    evaluation_results = {}
    for _ in range(num_evaluations):
        single_evaluation = evaluate_with_gpt(all_summaries, original_query)
        for key, score in single_evaluation.items():
            if key not in evaluation_results:
                evaluation_results[key] = []
            evaluation_results[key].append(score)

    avg_evaluation_results = {key: np.mean(scores) for key, scores in evaluation_results.items()}

    for subset_key, avg_score in avg_evaluation_results.items():
        print(f"summary_{subset_key}_avg_score: {avg_score:.2f}")

        subset = tuple(int(i) for i in subset_key)
        subset_value = avg_score
        cache[subset] = (subset_value, all_summaries[subset_key]['summary'])

        for i in subset:
            subset_without_i = tuple(j for j in subset if j != i)
            value_without_i = cache.get(subset_without_i, (0, ''))[0]
            marginal_contribution = subset_value - value_without_i
            shapley_values[i] += marginal_contribution * (math.factorial(len(subset)-1) * math.factorial(n-len(subset)) / math.factorial(n))

    full_summary_score = cache[tuple(range(n))][0]
    shapley_sum = sum(shapley_values)
    shapley_values = [value / shapley_sum * full_summary_score for value in shapley_values]

    print(f'\nTotal value = {full_summary_score:.2f}')
    for i in range(n):
        print(f"Shapley value for comment {i}: {shapley_values[i]:.2f}")

    return shapley_values