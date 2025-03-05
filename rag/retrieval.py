import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class RAGRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Ensure ChromaDB directory exists
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

        # Load Groq API Key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY is missing! Check your .env file.")

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    def retrieve_answer(self, user_query):
        document_chain = f"Use the retrieved documents to answer: {user_query}"
        return self.retriever.invoke({"input": document_chain})

rag_retriever = RAGRetriever()
