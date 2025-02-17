import dataclasses
from typing import List, Sequence, Iterator, Callable, Union

import lancedb
import ollama
import pyarrow as pa
from ollama import EmbedResponse, ChatResponse
from semantic_text_splitter import TextSplitter


@dataclasses.dataclass
class ModelPrompt:
    role: str
    content: str

    def serialise(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclasses.dataclass
class TableContent:
    content: str
    vectors: List[float]

    def serialise(self) -> dict:
        return {"vectors": self.vectors, "content": self.content}


class OllamaEasyRag:
    table_name: str = "sample_data_table"
    vector_cols_count: int = 1024
    ollama_chat_model_name: str = "qwen2.5:3b"
    ollama_vectorise_model_name: str = "bge-m3"
    create_prompts: Callable[[str, str], List[ModelPrompt]] = None
    allow_insert_duplicate_content: bool = False

    def __init__(self, create_prompts: Callable[[str, str], List[ModelPrompt]],
                 db_path: str = "data/sample-lancedb",
                 vector_cols_count: int = 1024,
                 ollama_chat_model_name: str = "qwen2.5:3b",
                 ollama_vectorise_model_name: str = "bge-m3",
                 allow_insert_duplicate_content: bool = False):
        if create_prompts is None:
            raise Exception("create_prompt param is required and found missing from OllamaEasyRag(...)")
        self.db = lancedb.connect(db_path)
        self.vector_cols_count = vector_cols_count
        self.ollama_chat_model_name = ollama_chat_model_name
        self.ollama_vectorise_model_name = ollama_vectorise_model_name
        self.create_prompts = create_prompts
        self.allow_insert_duplicate_content = allow_insert_duplicate_content

    def initialise(self) -> None:
        """
        1. creates table if it doesn't exist.
        2.
        :return: None
        """
        self.create_table()

    def create_table(self):
        schema = pa.schema(
            [
                pa.field("vectors", pa.list_(pa.float32(), list_size=self.vector_cols_count)),
                pa.field("content", pa.string())
            ])
        self.db.create_table(self.table_name, schema=schema, exist_ok=True)

    def insert_data(self, content: str, skip_duplicates=True, chunk_size: int = 1000,
                    splitter: Callable[[str], List[str]] = None) -> None:
        """
        Inserts provided data in the database.
        :param splitter: (content) -> List[str] a function that splits long text content in multiple small chunks
        :param chunk_size: Our default text splitter uses chunk_size to split text. max length of text chunk = chunk_size
        :param skip_duplicates: Should we ignore insert incase
        :param content: Text content to insert in database
        :return: None
        """

        # split text in multiple chunks
        default_splitter = TextSplitter(capacity=chunk_size)
        chunks = default_splitter.chunks(content) if splitter is None else splitter(content)

        # compute vectors for each chunk and open table
        items = [TableContent(content=chunk, vectors=self.compute_vectors(chunk)).serialise() for chunk in chunks]
        tbl = self.db.open_table(self.table_name)

        if skip_duplicates:
            tbl.merge_insert("content").when_matched_update_all().when_not_matched_insert_all().execute(items)
        else:
            tbl.add(items)

    def compute_vectors(self, content: str) -> Union[List[float], Sequence[float]]:
        """
        Computes vectors for provided query.
        :param content: Content to use for vectorisation
        :return: vectorised list of float values are returned.
        """
        response: EmbedResponse = ollama.embed(
            model=self.ollama_vectorise_model_name,
            input=content,
            truncate=False
        )
        return response.embeddings[0]

    def complete(self, prompts: List[ModelPrompt], stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Answers the provided prompt.

        :param prompts: Prompt taken by ML model to generate output
        :param stream: should the response be streamed?
        :return: Answer to the provided prompt.
        """
        response: Union[ChatResponse, Iterator[ChatResponse]] = ollama.chat(
            model=self.ollama_chat_model_name,
            messages=[prompt.serialise() for prompt in prompts],
            stream=stream
        )

        if not stream:
            yield response.message.content
        else:
            for chunk in response:
                yield chunk.message.content

    def search(self, query: str, stream: bool = False) -> Union[str, Iterator[str]]:
        """
        1. Perform vector search based for query
        2. Generates the output via AI model using results from step 1 as context.

        :param stream: Should the response be streamed or plain text response must be returned?
        :param query: Ask a question that needs to be answered based on RAG
        :return: Answer to the query post RAG
        """

        # compute vectors and search in database
        query_vectors = self.compute_vectors(query)
        table = self.db.open_table(self.table_name)
        context = table.search(query_vectors).limit(5).select(["content"]).to_list()
        context = "\n".join([record["content"] for record in context])

        # Generate model output based on context
        result = self.complete(prompts=self.create_prompts(context, query), stream=stream)
        return result if stream else list(result)[0]
