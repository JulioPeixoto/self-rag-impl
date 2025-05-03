from typing import Optional, Dict, Any
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class SelfRAGAgent:
    """
    Self-RAG Agent for retrieval augmented generation.
    
    This agent uses a vector database to store and retrieve documents,
    and a language model to generate responses based on retrieved context.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = OPENAI_API_KEY,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0,
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the Self-RAG agent.
        
        Args:
            openai_api_key: The OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model_name: The name of the language model to use.
            temperature: The temperature for the language model.
            embedding_model: The name of the embedding model to use.
        """
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model = embedding_model
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model=self.embedding_model
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        self.vector_store = None
        
        self.prompt_template = PromptTemplate(
            template="""Answer the question based on the context below.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
    
    def fetch_webpage(self, url: str) -> str:
        """
        Fetch content from a webpage.
        
        Args:
            url: The URL of the webpage to fetch.
            
        Returns:
            The text content of the webpage.
        """
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        
        return text
    
    def ingest_webpage(self, url: str) -> None:
        """
        Ingest a webpage into the vector store.
        
        Args:
            url: The URL of the webpage to ingest.
        """
        content = self.fetch_webpage(url)
        
        texts = self.text_splitter.split_text(content)
        
        docs = [Document(page_content=t, metadata={"source": url}) for t in texts]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)
    
    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Query the agent with a question.
        
        Args:
            query: The question to ask.
            k: The number of documents to retrieve.
            
        Returns:
            A dictionary containing the answer and relevant information.
        """
        if self.vector_store is None:
            return {"answer": "No documents have been ingested. Please ingest a webpage first."}
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
        
        result = qa_chain({"query": query})
        
        return {
            "query": query,
            "answer": result["result"],
            "retrieved_documents": k
        } 