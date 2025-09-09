import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import PDF_PATH, CHROMA_DIR, CHROMA_COLLECTION, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class ChromaPDFStore:
    def __init__(self):
        """Initializes the vector store."""
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.loader = PyPDFLoader(PDF_PATH)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        self.vs = self._get_vector_store()

    def _get_vector_store(self):
        """
        Loads the vector store from disk if it exists, otherwise creates it.
        """
        if os.path.exists(CHROMA_DIR):
            print("[Chroma] Loading existing vector store from disk...")
            return Chroma(
                persist_directory=CHROMA_DIR,
                collection_name=CHROMA_COLLECTION,
                embedding_function=self.embedding_model
            )
        else:
            print("[Chroma] Vector store not found. Creating and persisting new store...")
            pages = self.loader.load()
            docs = self.text_splitter.split_documents(pages)
            for i, doc in enumerate(docs):
                doc.metadata["chunk_id"] = i
            
            return Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                collection_name=CHROMA_COLLECTION,
                persist_directory=CHROMA_DIR
            )
            
    def similarity_search(self, query: str, k: int = 4):
        """Performs a similarity search."""
        return self.vs.similarity_search(query, k=k)

    def add_documents(self, docs):
        """Adds documents to the vector store."""
        self.vs.add_documents(docs)

    def delete_collection(self):
        """Deletes the entire collection."""
        self.vs.delete_collection()
        print(f"[Chroma] Collection '{CHROMA_COLLECTION}' deleted.")