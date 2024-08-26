import os
from typing import List
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from chromadb import HttpClient, Settings
from dotenv import load_dotenv
from utils import format_docs
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_transformers import DoctranQATransformer
import json
from langchain.storage.in_memory import InMemoryStore

load_dotenv()

# chat_history = []
chat_history = InMemoryStore()

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

    def transform(self, docs):
        qa_transformer = DoctranQATransformer(openai_api_model='gpt-3.5-turbo')
        transformed_document = qa_transformer.transform_documents(docs)
        return transformed_document

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

        vectordb.delete(ids=['123'])

        retriever = vectordb.as_retriever()

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # # TODO: add reference to source document
        # prompt = (
        #     {"context": retriever | format_docs, "input": RunnablePassthrough(), "chat_history": chat_history}
        #     | qa_system_prompt
        #     | self.llm
        #     | StrOutputParser()
        # )
        # print(prompt)
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # print(rag_chain)
        # chat_history.extend([HumanMessage(content=query), result["answer"]])
        chat_history.amset([HumanMessage(content=query), result["answer"]])

        return result["answer"]

def run_embedding():
    rag = RAG()

    # docs = rag.load()
    # print(f"Loaded {len(docs)} docs")

    # transformed_document = rag.transform(docs)
    # print(json.dumps(transformed_document, indent=2))

    # chunks = rag.split(docs)
    # print(f"Split into {len(chunks)} chunks")

    # vector_database = rag.store(chunks)
    # print(f"Stored in {vector_database}")

    # result = rag.chain("What is RDA in Mitrais?")
    # print(result)

if __name__ == "__main__":
    run_embedding()