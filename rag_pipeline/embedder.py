from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    """
    A wrapper class for loading a multilingual embedding model
    to be used in vector stores or LLM pipelines.

    Default model: HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1
    """

    def __init__(self, model_name: str = "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The Hugging Face model name to use for embeddings.
        """
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def get(self):
        """
        Returns the LangChain-compatible embedding model instance.

        Returns:
            HuggingFaceEmbeddings: The embedding model.
        """
        return self.embedding_model
