from fastapi import FastAPI, Query
from contextlib import asynccontextmanager

from rag_pipeline.prompts import get_rent_law_prompt
from rag_pipeline.embedder import Embedder
from rag_pipeline.vectordb import VectorDB
from rag_pipeline.chain import RentLawQAChain

# Globals to hold RAG components
vector_store = None
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, rag_chain

    embedder = Embedder()
    embedding_model = embedder.get()

    vector_store = VectorDB(embedding_model=embedding_model, add_new_table=False)
    await vector_store.setup()

    retriever = vector_store.get_retriever()
    prompt_template = get_rent_law_prompt()
    rag_chain = RentLawQAChain(
        model_name="gpt-4o", retriever=retriever, prompt_template=prompt_template
    )

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/query")
async def query(text: str = Query(..., description="The user's question")):
    answer = rag_chain.ask(text)
    return {"answer": answer}
