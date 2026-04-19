from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents; metadata kept on chunks."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class TokenChunker(BaseChunker):
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


def get_chunker(strategy: str = "recursive") -> BaseChunker:
    if strategy == "recursive":
        return RecursiveCharacterChunker()
    elif strategy == "token":
        return TokenChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use 'recursive' or 'token'.")
