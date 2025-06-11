import math
import numpy as np
from itertools import combinations
from src.llm_summarization import (
    label_comment_relevance
)

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