from typing import List

from src.ollama_easy_rag import OllamaEasyRag, ModelPrompt


def test_instantiates():
    def prepare_prompt(context: str, query: str) -> List[ModelPrompt]:
        return [
            ModelPrompt(role="assistant", content="You should be smart!"),
            ModelPrompt(role="user", content=f"Query: {query},  Context: {context}")
        ]

    assert OllamaEasyRag(create_prompts=prepare_prompt) is not None
