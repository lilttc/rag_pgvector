# cli.py
import argparse
import asyncio
from rag_pipeline.embedder import Embedder
from rag_pipeline.vectordb import VectorDB
from rag_pipeline.prompts import get_rent_law_prompt
from rag_pipeline.chain import RentLawQAChain

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    embedder = Embedder()
    embedding_model = embedder.get()
    db = VectorDB(embedding_model=embedding_model, add_new_table=False)
    await db.setup()
    prompt_template = get_rent_law_prompt()
    retriever = db.get_retriever()
    rag_chain = RentLawQAChain(model_name="gpt-4o", retriever=retriever, prompt_template=prompt_template)
    answer = rag_chain.ask(args.query)
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
