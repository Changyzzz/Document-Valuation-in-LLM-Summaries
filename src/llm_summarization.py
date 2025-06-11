import openai
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