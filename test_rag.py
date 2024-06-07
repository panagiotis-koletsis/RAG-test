from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_zelensky_false():
    assert not query_and_validate(
        question="who is zelensky a) Ukrainian president b) american president c) ex greek vlogger",
        expected_response="a) ex greek vlogger",
    )

def test_zelensky():
    assert query_and_validate(
        question="who is zelensky a) Ukrainian president b) american president c) ex greek vlogger",
        expected_response="a) Ukrainian president",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="What percentage of votes did zelensky had in order to get elected in the second round? a)5% b)73% c)50% d)30%",
        expected_response="b) 73%",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )