from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

class RentLawQAChain:
    """
    Builds a Retrieval-Augmented Generation (RAG) chain to answer questions about
    Dutch rental law using an LLM and a vector store retriever.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt_template: PromptTemplate,
        model_name: str = "gpt-4o",
        temperature: float = 0.0
    ):
        """
        Initialize the QA chain.

        Args:
            retriever (VectorStoreRetriever): Retriever to fetch relevant context.
            prompt_template (PromptTemplate): Prompt structure used to guide the LLM.
            model_name (str): OpenAI model name (e.g. 'gpt-4o').
            temperature (float): Sampling temperature for creativity (0 = deterministic).
        """
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query: str) -> str:
        """
        Run a question through the RAG chain and return the generated answer.

        Args:
            query (str): A natural language question about Dutch rental law.

        Returns:
            str: The LLM-generated answer based on retrieved context.
        """
        return self.chain.invoke(query)
