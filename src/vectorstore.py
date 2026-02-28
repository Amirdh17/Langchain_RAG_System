import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingPipeline
from data_loader import load_all_documents
from data_loader import move_all_files

class FaissVectorStore:
    """  
        Build a vector store in specified path 
    """
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """  
            This constructor initialize the required variables and embedding model
        """
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._initialize()
        
    def _initialize(self) -> None:
        """  
            Initialize directory path and load model for embedding purpose
        """
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            print(f"[INFO] Loading embedding model: {self.embedding_model}")
            self.model = SentenceTransformer(self.embedding_model)
            print(f"[INFO] Successfully! Loaded embedding model: {self.embedding_model}")
        except Exception as e:
            print(f"[DEBUG] Error during loading embedding model: {e}")
    
    def build_from_documents(self, documents: List[Any]):
        """  
            Loads the vector for given documents into vector store
            Args:
                documents: List[Any] -> List of documents to embed and store them into vector store 
        """
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        # print(f"[INFO] Chunks : {chunks}")
        metadatas = []
        for chunk in chunks:
            metadatas.extend([{"text": doc.page_content, "source":doc.metadata.get('source_file'), "page_number": doc.metadata.get('page')} for doc in chunk])
        print(f"[INFO] Metadata : {metadatas}")
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None) -> None:
        """  
            populate Index with embeddings (vectors) and matadata with metadatas (raw chunks)
            Args:
            embeddings: np.ndarray -> embeddings of all chunks
            metadatas: List[Any] -> raw chunks
        """
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        """  
            store index (vectors) using write_index and store metadata (raw chunks) using pickle's serialization (byte stream)
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        """  
            Loads Index and Raw chunks into self.index & self.metadata from locally stored in directory
        """
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Any]:
        """  
            Search related chunks using embedded query
            Args:
                query_embedding: np.ndarray -> user's query in vector
                top_k: int -> top k matching results
            Return:
                results: List[Any] -> List of Dict of index, distance and metadata which has the content & metadata 
        """
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        """  
            Coverts given query text into vector query and calls search function
            Args:
                query_text: str -> user's query in string
                top_k: int -> Top k matching results
            Returns:
                results: List[Any] -> List of Dict of index, distance and metadata which has the content & metadata 
        """
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
        
if __name__ == "__main__":
    docs = load_all_documents("data")

    # Once loaded, moving all files to archive
    # move_all_files("../data", "../archive")

    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is Money Market?", top_k=3))
    # store = FaissVectorStore("faiss_store")
    # store.load()
    # print(store.metadata)

