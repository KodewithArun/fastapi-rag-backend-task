"""
Document Parsing Service

This module abstracts the file parsing logic for the Document Ingestion API.
It relies on an Abstract Base Class (ABC) to ensure all parsers follow the same interface.
"""
from abc import ABC, abstractmethod
from typing import BinaryIO
import pymupdf

class BaseDocumentParser(ABC):
    """
    Abstract Base Class for all document parsers.
    Forces all implementing classes to provide an extract_text method.
    """
    @abstractmethod
    def extract_text(self, file: BinaryIO) -> str:
        """
        Extracts raw text from a binary file stream.
        
        Args:
            file (BinaryIO): The uploaded file byte stream.
            
        Returns:
            str: The extracted text context.
        """
        pass

class TXTParser(BaseDocumentParser):
    """Implementation of document parser for standard text files."""
    
    def extract_text(self, file: BinaryIO) -> str:
        # Read the binary file directly and decode as UTF-8
        return file.read().decode("utf-8", errors="replace")

class PDFParser(BaseDocumentParser):
    """Implementation of document parser for PDF files using PyMuPDF (fitz)."""
    
    def extract_text(self, file: BinaryIO) -> str:
        try:
            # PyMuPDF expects bytes buffer to read from streams natively
            file_bytes = file.read()
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            
            text_blocks = []
            for page in doc:
                extracted = page.get_text()
                if extracted:
                    text_blocks.append(extracted)
                    
            doc.close()
            return "\n".join(text_blocks)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file: {str(e)}")

def get_document_parser(content_type: str) -> BaseDocumentParser:
    """
    Factory function to retrieve the correct parser based on the file MIME type.
    
    Args:
        content_type (str): The MIME type of the uploaded file.
        
    Returns:
        BaseDocumentParser: An instantiated parser object.
    
    Raises:
        ValueError: If the file type is entirely unsupported.
    """
    if content_type == "text/plain":
        return TXTParser()
    elif content_type == "application/pdf":
        return PDFParser()
    else:
        raise ValueError(f"Unsupported document format: {content_type}. Only .txt and .pdf are allowed.")
