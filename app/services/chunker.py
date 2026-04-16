"""
Chunking Service

Provides multiple text chunking strategies to split large documents into
smaller, more manageable embeddings payloads.
Uses Abstract Base Classes.
"""
from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


class BaseChunker(ABC):
    """Abstract Base Class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Splits the given text into an array of smaller strings.
        
        Args:
            text (str): The raw text to split.
            
        Returns:
            List[str]: A list of text chunks.
        """
        pass

class RecursiveCharacterChunker(BaseChunker):
    """
    Standard Strategy: Splits text progressively down a list of characters
    (paragraphs, sentences, words) until the chunk size is met.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

class TokenChunker(BaseChunker):
    """
    Alternative Strategy: Splits exact token lengths using OpenAI's tiktoken.
    This provides highly predictable token counts for the embedding model.
    """
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

def get_chunker(strategy: str = "recursive") -> BaseChunker:
    """
    Factory function to retrieve the required chunking strategy.
    
    Args:
        strategy (str): 'recursive' or 'token'
        
    Returns:
        BaseChunker: The instantiated chunker interface.
    """
    if strategy == "recursive":
        return RecursiveCharacterChunker()
    elif strategy == "token":
        return TokenChunker()
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use 'recursive' or 'token'.")
