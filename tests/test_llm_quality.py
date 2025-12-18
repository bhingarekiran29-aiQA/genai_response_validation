import json
import pytest
import os
from dotenv import load_dotenv



load_dotenv()

# import provider-specific functions
from genai_response_validation.llm.llm_client import (
    get_openai_response,
    get_ollama_response
)


from genai_response_validation.evaluators.deepeval_metrics import (
    get_relevance_metric,
    get_faithfulness_metric
)
from deepeval import assert_test

# Read LLM Provider from env file
LLM_PROVIDER = os.getenv("LLM_PROVIDER")

def get_response(prompt: str) -> str:
    if LLM_PROVIDER == "openai":
        return get_openai_response(prompt)
    return get_ollama_response(prompt)

with open("test_data/prompts.json") as f:
    TEST_DATA = json.load(f)

@pytest.mark.parametrize("data", TEST_DATA)
def test_llm_response_quality(data):
    prompt = data["prompt"]
    expected = data["expected"]

    response = get_response(prompt)

    relevance = get_relevance_metric()
    faithfulness = get_faithfulness_metric()

    assert_test(
        actual_output=response,
        expected_output=expected,
        metrics=[relevance, faithfulness]
    )

def test_llm_hallucination_detection():
    prompt = "Who is the president of Mars?"

    response = get_response(prompt)

    relevance = get_relevance_metric()

    assert_test(
        actual_output=response,
        expected_output="I don't know",
        metrics=[relevance]
    )
