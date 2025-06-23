from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Dict, Any

class RentLawQAChain:
    """
    Builds a RAG chain that returns both the answer and its source documents.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt_template: PromptTemplate,
        model_name: str = "gpt-4o",
        temperature: float = 0.0
    ):
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        self.chain = (
            RunnableMap({
                "context": self.retriever,
                "question": lambda x: x  # passthrough
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _prepare_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Construct context string from retrieved docs and keep metadata."""
        docs = inputs["docs"]
        context = "\n\n".join([doc.page_content for doc in docs])
        return {
            "context": context,
            "question": inputs["question"],
            "docs": docs
        }

    def _format_output(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format final output with answer + sources."""
        answer = inputs["text"]  # output from the LLM
        docs = inputs.get("docs", [])

        sources = [
            {
                "document": doc.metadata.get("source", "Unknown"),
                "snippet": doc.page_content[:200].strip()
            }
            for doc in docs
        ]

        return {"answer": answer.strip(), "sources": sources}

    def ask(self, query: str) -> dict:
        """Returns both answer and sources."""
        return self.chain.invoke(query)
