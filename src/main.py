import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv


from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

# from langchain_openai import OpenAIEmbeddings

from database import Database
from query_data import query_rag
from tools.pdf_document_extractor import PdfDocumentExtractor

test_data = "/workspaces/rag-chromadb/data/"
db_path = "/workspaces/rag-chromadb/db"
load_dotenv()

pdfs_extractor = PdfDocumentExtractor()
docs = pdfs_extractor.get_documents(test_data)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set")

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY is not set")


embedding = OpenAIEmbeddings()
db = Database(db_path, embedding)
inserted = db.upsert_chunks(docs)

print(f"Inserted {inserted} document(s)")

chat = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192", temperature=0.3)  # type: ignore
name = input("Qual é o seu nome? ")
question = input(f"Faça sua pergunta, {name}: ")
response = query_rag(question, name, db, chat)

print(response)
