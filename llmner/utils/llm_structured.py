from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Any


# Define a structured model for the LLM response
class LLMResponse(BaseModel):
    ner_entities: dict = Field(default={}, description="The named entities, "
                                                       "e.g. {'NAME': ['John Doe', 'John'], 'ADDRESS': ['New York']}")


class LLMHandler:
    def __init__(self, api: ChatOpenAI, prompt_template: ChatPromptTemplate):
        self.api = api.with_structured_output(LLMResponse)
        self.prompt_template = prompt_template

    def request_llm(self, query: str) -> LLMResponse:
        """
        Request the LLM and return a structured response.

        :param query: The query to send to the LLM
        :return: A structured LLMResponse object
        """

        def ask_api(api: Any, query: str, prompt: str) -> LLMResponse:
            """
            Internal function to invoke the API and parse the response.

            :param api: The API object to use for the query
            :param query: The query string
            :param prompt: The prompt template to format the query
            :return: A parsed LLMResponse object
            """
            # Create a chain of operations
            chain = prompt | api
            message = chain.invoke({"query": query})

            # Assuming the response can be validated using the LLMResponse model
            try:
                structured_response = LLMResponse.parse_raw(message.content)
            except Exception as e:
                print(f"Failed to parse response: {e}")
                # Return a fallback response in case of parsing failure
                structured_response = LLMResponse(
                    ner_entities={}
                )

            return structured_response

        # Create the user-specific prompt
        prompt = self.prompt_template.format(query=query)

        # Log the request process
        print("Requesting LLM...")

        # Call the API with the query and the prompt
        structured_message = ask_api(self.api, query, prompt)
        return structured_message
