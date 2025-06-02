from langchain_postgres import PGEngine, PGVectorStore, Column
from sqlalchemy.exc import ProgrammingError

class VectorDB:
    """
    A class to manage the connection and setup of a pgvector-backed vector store
    for use in RAG pipelines.
    """

    def __init__(
        self,
        embedding_model,
        add_new_table: bool = False,
        table_name: str = "wet_betaalbare_huur_act",
        vector_size: int = 896,
        db_user: str = "my_pg_user",
        db_password: str = "pg_vector",
        db_host: str = "localhost",
        db_port: str = "5432",
        db_name: str = "vector_db",
    ):
        """
        Initialize the VectorDB configuration and connection.

        Args:
            embedding_model: LangChain-compatible embedding model (e.g., HuggingFaceEmbeddings).
            add_new_table (bool): Whether to create a new table in the vector store.
            table_name (str): Name of the vector table in PostgreSQL.
            vector_size (int): Size of the embedding vector.
            db_user, db_password, db_host, db_port, db_name: DB connection settings.
        """
        self.embedding_model = embedding_model
        self.add_new_table = add_new_table
        self.table_name = table_name
        self.vector_size = vector_size
        self.connection_string = (
            f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

    async def setup(self):
        """
        Create engine, initialize table (if needed), and return an active vector store.
        
        Returns:
            PGVectorStore: An initialized vector store instance ready for use.
        """
        self.pg_engine = PGEngine.from_connection_string(url=self.connection_string)
        if self.add_new_table == True:
            try:
                await self.pg_engine.ainit_vectorstore_table(
                    table_name=self.table_name,
                    vector_size=self.vector_size,
                    id_column="langchain_id",
                    content_column="content",
                    embedding_column="embedding",
                    metadata_columns=[
                        Column("source", "TEXT")
                    ],
                )
            except ProgrammingError:
                print(f"Table '{self.table_name}' already exists. Skipping creation.")

        self.vector_store = await PGVectorStore.create(
            engine=self.pg_engine,
            table_name=self.table_name,
            embedding_service=self.embedding_model,
        )

        return self.vector_store
    
    async def add_documents(self, documents):
        """
        Add documents to the initialized vector store.

        Args:
            documents (list[Document]): List of LangChain documents to embed and insert.
        """
        if not hasattr(self, "vector_store"):
            raise RuntimeError("Vector store not initialized. Call await setup() first.")
        await self.vector_store.aadd_documents(documents)

    def get_retriever(self, k: int = 3):
        """
        Returns a retriever object from the vector store.

        Args:
            k (int): Number of similar documents to retrieve.

        Returns:
            BaseRetriever: LangChain-compatible retriever object.
        """
        if not hasattr(self, "vector_store"):
            raise RuntimeError("Vector store not initialized. Call await setup() first.")
        return self.vector_store.as_retriever(search_type="similarity", k=k)

