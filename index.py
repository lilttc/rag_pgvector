# index.py
import asyncio
from rag_pipeline.loader import PDFLoader
from rag_pipeline.splitter import TextSplitter
from rag_pipeline.embedder import Embedder
from rag_pipeline.vectordb import VectorDB

async def main():
    loader = PDFLoader("documents")
    raw_docs = loader.load_all_pdf_files()

    splitter = TextSplitter()
    split_docs = splitter.split(raw_docs)

    embedder = Embedder()
    embedding_model = embedder.get()
    db = VectorDB(embedding_model=embedding_model, add_new_table=True)
    await db.setup()
    await db.add_documents(split_docs)

if __name__ == "__main__":
    asyncio.run(main())
