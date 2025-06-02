import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TextSplitter:
    """
    A utility class to clean and split documents into smaller chunks using 
    LangChain's RecursiveCharacterTextSplitter.

    Suitable for use with LLM pipelines where chunking improves retrieval quality.
    """

    def __init__(self, chunk_size=2000, chunk_overlap=200, separators=None):
        """
        Initialize the text splitter.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
            separators (list[str], optional): Custom separators for splitting text.
                Defaults to ["\n\n", "\n", ".", " "].
        """
        if separators is None:
            separators = ["\n\n", "\n", ".", " "]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Clean and split a list of documents into smaller chunks.

        Prepends article headers (e.g., 'Artikel 3') to improve semantic structure.

        Args:
            documents (list[Document]): A list of LangChain Document objects.

        Returns:
            list[Document]: A list of smaller, chunked Document objects.
        """
        cleaned_docs = []

        for doc in documents:
            content = doc.page_content
            header_match = re.search(r"(Artikel\s+\w+)", content)
            if header_match:
                content = f"{header_match.group(1)}\n{content}"
            cleaned_docs.append(Document(page_content=content, metadata=doc.metadata))

        return self.splitter.split_documents(cleaned_docs)
