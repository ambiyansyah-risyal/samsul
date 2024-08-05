import os
from typing import List
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain import hub
from chromadb import HttpClient, Settings
from dotenv import load_dotenv
from utils import format_docs
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAG:
    def __init__(self):
        self.bucket = os.getenv("LOADER_S3_BUCKET")
        self.aws_access_key_id = os.getenv("LOADER_AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("LOADER_AWS_SECRET_ACCESS_KEY")
        self.endpoint_url = os.getenv("LOADER_ENDPOINT_URL")
        self.use_ssl=os.getenv("LOADER_USE_SSL")
        self.chroma_db_host = os.getenv("CHROMA_DB_HOST")
        self.chroma_db_port = os.getenv("CHROMA_DB_PORT")
        self.chroma_db_collection_name = os.getenv("CHROMA_DB_COLLECTION_NAME")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=100,
            chunk_overlap=0,
        )
        self.llm = ChatOpenAI()
        self.embedding_model = OpenAIEmbeddings()

    def load(self) -> List[Document]:
        loader = S3DirectoryLoader(
            bucket=self.bucket,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
            use_ssl=self.use_ssl,
        )

        return loader.load()

    def split(self, docs: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(docs)

    def store(self, documents: List[Document]):
        client = HttpClient(
            host=self.chroma_db_host,
            port=self.chroma_db_port,
            settings=Settings(allow_reset=True),
        )
        client.reset()

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            client=client,
            collection_name=self.chroma_db_collection_name,
        )

        return vectorstore

    def chain(self, query: str):
        prompt = hub.pull("rlm/rag-prompt")

        client = HttpClient(
            host=self.chroma_db_host,
            port=self.chroma_db_port,
            settings=Settings(allow_reset=True),
        )

        vectordb = Chroma(
            client=client,
            collection_name=self.chroma_db_collection_name,
            embedding_function=self.embedding_model,
        )

        retriever = vectordb.as_retriever()

        # TODO: add reference to source document
        prompt = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return prompt.stream(query)

def run_embedding():
    rag = RAG()

    docs = rag.load()
    print(f"Loaded {len(docs)} docs")

    chunks = rag.split(docs)
    print(f"Split into {len(chunks)} chunks")

    vectordb = rag.store(chunks)
    print(f"Stored in {vectordb}")

if __name__ == "__main__":
    run_embedding()