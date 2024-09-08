from dataclasses import dataclass
from typing import Callable
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class TextSplittingOptions:
    separator: str = "\n"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    length_function: Callable = len
    is_separator_regex: bool = False


class BaseDocumentExtractor:
    pass


class PdfDocumentExtractor(BaseDocumentExtractor):
    def __init__(
        self,
        options: TextSplittingOptions = TextSplittingOptions(),
    ):

        self._text_splitter = RecursiveCharacterTextSplitter(
            # separator=options.separator,
            chunk_size=options.chunk_size,
            chunk_overlap=options.chunk_overlap,
            length_function=options.length_function,
            is_separator_regex=options.is_separator_regex,
        )

    def _split_documents(self, docs: list[Document]) -> list[Document]:
        return self._text_splitter.split_documents(docs)

    def get_documents(self, docs_root_path: str) -> list[Document]:
        docs = PyPDFDirectoryLoader(docs_root_path).load()
        return self._split_documents(docs)
