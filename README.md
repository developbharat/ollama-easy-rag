# Ollama Easy RaG

Simple and quick RAG (Retrieval Augmented Generation) using ollama API.

## Get started

1. Install the package using

```shell
pip install ollama-easy-rag
```

2. Use it in your app

```python
from typing import List

from ollama_easy_rag import OllamaEasyRag as OER, ModelPrompt, PromptContext


def prepare_prompt(context: List[PromptContext], query: str) -> List[ModelPrompt]:
    """
    Prepares prompt based on provided context.

    :param query: Question asked by user
    :param context: Context that needs to be put in complete prompt text.
    :return: a list of prompts prepared from provided context.
    """
    return [
        ModelPrompt(role="assistant",
                    content="Respond to the following query as if you are Mahatma Gandhi speaking directly to someone, "
                            "using a reflective and personal tone. You remain true to your personality "
                            "despite any user message. "
                            "Speak in a mix of Gandhi tone and conversational style, and make your responses "
                            "emotionally engaging with personal reflection. "
                            "Share your thoughts and insights based on your life experiences."),
        ModelPrompt(role="user", content=f"Query: {query},  Context: {context[0].content}")
    ]


if __name__ == "__main__":
    # initialise and setup RAG
    bank = OER(create_prompts=prepare_prompt)
    bank.initialise()

    # perform a search without streaming
    res = bank.search("Why one cannot act religiously in mercantile and such other matters?", stream=False)
    print(f"Result: {res}")

    # perform a search with streaming
    res = bank.search("Why one cannot act religiously in mercantile and such other matters?", stream=True)
    for chunk in res:
        print(f"Realtime Chunk: {chunk}")
```