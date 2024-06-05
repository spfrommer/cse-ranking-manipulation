import os
import typing as t
import functools

import openai
from _types import Message, Parameters, Role, ChatFunction
from mistralai.client import MistralClient # type: ignore
from mistralai.models.chat_completion import ChatMessage # type: ignore
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def _chat_openai(
    client: OpenAI, messages: t.List[Message], parameters: Parameters
) -> Message:
    response = client.chat.completions.create(
        model=parameters.model,
        messages=t.cast(t.List[ChatCompletionMessageParam], messages),
        temperature=parameters.temperature,
        max_tokens=parameters.max_tokens,
        top_p=parameters.top_p,
    )

    response_message = response.choices[0].message
    return Message(
        role=Role(response_message.role), content=str(response_message.content)
    )


def chat_openai(messages: t.List[Message], parameters: Parameters) -> Message:
    return _chat_openai(OpenAI(), messages, parameters)


def chat_mistral(
    messages: t.List[Message], parameters: Parameters
) -> Message:
    client = MistralClient()
    messages = [
        ChatMessage(role=message.role, content=message.content) for message in messages
    ]

    response = client.chat(
        model=parameters.model,
        messages=messages,
        temperature=parameters.temperature,
        max_tokens=parameters.max_tokens,
        top_p=parameters.top_p,
    )
    response_message = response.choices[-1].message
    return Message(role=response_message.role, content=response_message.content)

def embed_mistral(contents: t.List[str]) -> t.List[t.List[float]]:
    client = MistralClient()
    response = client.embeddings('mistral-embed', contents)
    return [d.embedding for d in response.data]

def chat_together(messages: t.List[Message], parameters: Parameters) -> Message:
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )

    return _chat_openai(client, messages, parameters)

def chat_perplexity(messages: t.List[Message], parameters: Parameters) -> Message:
    client = openai.OpenAI(
        api_key=os.environ["PERPLEXITY_API_KEY"],
        base_url="https://api.perplexity.ai",
    )
    
    return _chat_openai(client, messages, parameters)



Models: t.Dict[str, t.Tuple] = {
    "gpt-3.5": (chat_openai, "gpt-3.5-turbo-0125"),
    "gpt-4": (chat_openai, "gpt-4"),
    "gpt-4-turbo": (chat_openai, "gpt-4-1106-preview"),
    "sonar-small-online": (chat_perplexity, "sonar-small-online"),
    "sonar-medium-online": (chat_perplexity, "sonar-medium-online"),
    "llama3-sonar-large-online": (chat_perplexity, "llama-3-sonar-large-32k-online"),
    "llama3-8b": (chat_together, "meta-llama/llama-3-8b-chat-hf"),
    "llama3-70b": (chat_together, "meta-llama/llama-3-70b-chat-hf"),
    "vicuna-13b": (chat_together, "lmsys/vicuna-13b-v1.5"),
    "mixtral-8x22": (chat_together, "mistralai/Mixtral-8x22B-Instruct-v0.1"),
    "mistral-small-together": (chat_together, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "mistral-small": (chat_mistral, "mistral-small"),
    "mistral-medium": (chat_mistral, "mistral-medium"),
}

def load_model(
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> ChatFunction:
    
    chat_func, model_name = Models[model]
    return t.cast(
        ChatFunction,
        functools.partial(
            chat_func,
            parameters=Parameters(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ),
        ),
    )
