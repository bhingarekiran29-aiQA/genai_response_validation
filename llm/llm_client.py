from openai import OpenAI
import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL"),
        messages=[
            {"role": "user", "content": prompt}
            ],
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

def get_ollama_response(prompt: str) -> str:
    llm = ChatOllama(
        base_url= os.getenv("OLLAMA_API_URL"),
        model= os.getenv("OLLAMA_MODEL"),
        temperature= float(os.getenv("OLLAMA_TEMPERATURE")),
        max_tokens= int(os.getenv("OLLAMA_MAX_TOKENS"))
    )

    response = llm.invoke(prompt)
    return response.content


def get_llm_response(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        return get_openai_response(prompt)
    return get_ollama_response(prompt)