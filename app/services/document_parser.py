from abc import ABC, abstractmethod
from typing import BinaryIO

import pymupdf


class BaseDocumentParser(ABC):
    @abstractmethod
    def extract_text(self, file: BinaryIO) -> str:
        pass

class TXTParser(BaseDocumentParser):
    def extract_text(self, file: BinaryIO) -> str:
        return file.read().decode("utf-8", errors="replace")

class PDFParser(BaseDocumentParser):
    def extract_text(self, file: BinaryIO) -> str:
        try:
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
    if content_type == "text/plain":
        return TXTParser()
    elif content_type == "application/pdf":
        return PDFParser()
    else:
        raise ValueError(f"Unsupported document format: {content_type}. Only .txt and .pdf are allowed.")
