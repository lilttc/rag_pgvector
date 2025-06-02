from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class PDFLoader:
    """
    A utility class to load and preprocess PDF documents from a specified folder
    using LangChain's PyPDFLoader.

    Each loaded document includes metadata indicating the source filename.

    Attributes:
        folder_path (Path): The path to the folder containing PDF files.
    """

    def __init__(self, folder_path: str):
        """
        Initialize the PDFLoader with the path to a folder.

        Args:
            folder_path (str): Path to the folder containing PDF documents.

        Raises:
            FileNotFoundError: If the specified folder does not exist.
        """
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist")

    def load_all_pdf_files(self) -> list[Document]:
        """
        Load all PDF documents in the specified folder using PyPDFLoader.

        Adds metadata to each document indicating the source filename.

        Returns:
            list[Document]: A list of LangChain Document objects with content and metadata.
        """
        all_docs = []

        for file_path in self.folder_path.glob("*.pdf"):
            print(f"Loading: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = file_path.name

            all_docs.extend(documents)

        return all_docs
