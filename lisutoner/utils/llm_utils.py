import openai
import os
from loguru import logger
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_random


def create_langchain_prompt(user_prompt, mode="market_research"):
    """
    Available modes: market_research, creative_writing, general
    """
    prompts = {
        "market_research": "You are a market research assistant",
        "creative_writing": "You are a creative writer",
        "general": "You are a helpful chatbot",
    }
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts[mode]),
        ("human", user_prompt),
    ])
    return prompt


def ask_api(api, prompt, query, prompt_keyword):
    chain = prompt | api
    with get_openai_callback() as cb:
        message = chain.invoke({prompt_keyword: query})
    return message, cb.total_cost, cb.total_tokens


@retry(wait=wait_random_exponential(min=1, max=42), stop=stop_after_attempt(4))
def completion_with_backoff(**kwargs):
    return ask_api(**kwargs)


openai.api_key = os.getenv('OPENAI_API_KEY')
logger.info(f"\033[093mOpenAI API Key: {openai.api_key}")
logger.info(f"\033[093mOpenAI Model: {os.getenv('OPENAI_MODEL', 'n/a')}")

openai_llm = ChatOpenAI(temperature=os.getenv('TEMPERATURE', 0.),
                        max_tokens=os.getenv('MAX_TOKENS', 1500),
                        model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                        openai_api_key=os.getenv('OPENAI_API_KEY'))
