import dataclasses
from typing import List, Sequence, Iterator, Callable, Union, Optional, Tuple

import lancedb
import ollama
import pyarrow as pa
from ollama import EmbedResponse, ChatResponse


@dataclasses.dataclass
class PromptContext:
    source: str
    chunk_id: str
    content: str

    def serialise(self) -> dict:
        return {"source": self.source, "chunk_id": self.chunk_id, "content": self.content}


@dataclasses.dataclass
class ModelPrompt:
    role: str
    content: str

    def serialise(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclasses.dataclass
class SearchSource:
    source: str
    chunk_id: str

    def serialise(self):
        return {"source": self.source, "chunk_id": self.chunk_id}


@dataclasses.dataclass
class SearchResult:
    prediction: str
    sources: List[SearchSource]

    def serialise(self):
        return {"prediction": self.prediction, "sources": [source.serialise() for source in self.sources]}


@dataclasses.dataclass
class TableContent:
    content: str
    vectors: List[float]
    source: Optional[str]
    chunk_id: Optional[str]

    def serialise(self) -> dict:
        return {"vectors": self.vectors,
                "content": self.content,
                "source": self.source if self.source is not None else "",
                "chunk_id": self.chunk_id if self.chunk_id is not None else ""}


class OllamaEasyRag:
    table_name: str = "sample_data_table"
    vector_cols_count: int = 1024
    ollama_chat_model_name: str = "qwen2.5:3b"
    ollama_vectorise_model_name: str = "bge-m3"
    create_prompts: Callable[[List[PromptContext], str], List[ModelPrompt]] = None
    allow_insert_duplicate_content: bool = False

    def __init__(self, create_prompts: Callable[[List[PromptContext], str], List[ModelPrompt]],
                 db_path: str = "data/sample-lancedb",
                 table_name: str = "sample_data_table",
                 vector_cols_count: int = 1024,
                 ollama_chat_model_name: str = "qwen2.5:3b",
                 ollama_vectorise_model_name: str = "bge-m3",
                 allow_insert_duplicate_content: bool = False):
        if create_prompts is None:
            raise Exception("create_prompt param is required and found missing from OllamaEasyRag(...)")
        self.table_name = table_name
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
                pa.field("content", pa.string()),
                pa.field("source", pa.string()),
                pa.field("chunk_id", pa.string())
            ])
        self.db.create_table(self.table_name, schema=schema, exist_ok=True)

    def insert_data(self, data: List[TableContent], skip_duplicates=True) -> None:
        """
        Inserts provided data in the database.
        :param create_chunks: a function that splits long text content in multiple small chunks
        :param chunk_size: Our default text splitter uses chunk_size to split text. max length of text chunk = chunk_size
        :param skip_duplicates: Should we ignore insert incase
        :param data: content to insert in database
        :return: None
        """

        # open table
        tbl = self.db.open_table(self.table_name)
        serialised = [record.serialise() for record in data]

        if skip_duplicates:
            tbl.merge_insert("content").when_matched_update_all().when_not_matched_insert_all().execute(serialised)
        else:
            tbl.add(serialised)

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

    @staticmethod
    def __prepare_context(table: lancedb.table.Table,
                          search_vectors: Union[List[float], Sequence[float]],
                          search_limit: int) -> Tuple[List[PromptContext], List[SearchSource]]:
        context = table.search(search_vectors).limit(search_limit).select(["content", "source", "chunk_id"]).to_list()

        # Prepare context and sources
        sources: List[SearchSource] = []
        contexts: List[PromptContext] = []
        for record in context:
            content = record["content"]
            source = record["source"]
            chunk_id = record["chunk_id"]

            # Add context
            contexts.append(PromptContext(content=content,
                                          source=source if source is not None else "",
                                          chunk_id=chunk_id if chunk_id is not None else ""))
            # Add source
            sources.append(SearchSource(source=source if source is not None else "",
                                        chunk_id=chunk_id if chunk_id is not None else ""))

        return contexts, sources

    def search(self, query: str, stream: bool = False, search_limit: int = 5) -> SearchResult:
        """
        1. Perform vector search based for query
        2. Generates the output via AI model using results from step 1 as context.

        :param search_limit: No of records to search in database for context.
        :param stream: Should the response be streamed or plain text response must be returned?
        :param query: Ask a question that needs to be answered based on RAG
        :return: Answer to the query post RAG
        """

        # compute vectors and search in database
        query_vectors = self.compute_vectors(query)
        table = self.db.open_table(self.table_name)
        context, sources = self.__prepare_context(table=table, search_vectors=query_vectors, search_limit=search_limit)

        # Generate model output based on context
        result = self.complete(prompts=self.create_prompts(context, query), stream=stream)

        if stream:
            return SearchResult(prediction=result, sources=sources)
        else:
            return SearchResult(prediction=list(result)[0], sources=sources)
