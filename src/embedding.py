from typing import List, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from data_loader import load_all_documents

class EmbeddingPipeline:
    """
    This class handles document chunking and embedding them into vectors
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """  
            This constructor loads the given model and initialize the variables
            Args:
                model_name: str
                chunk_size: int
                chunk_overlap: int
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the sentence transformer model
        """
        try:
            print(f"[INFO] Loading Embedded model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"[INFO] Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        except Exception as e:
            print(f"[DEBUG] Error during loading model {self.model_name}: {e}")
            raise
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """  
        This function split the documents into small chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )
        chunks = [text_splitter.split_documents(doc) for doc in documents]

        # Count documents
        docs_cnt = 0
        for doc in documents:
            docs_cnt += len(doc)
        
        # Count total chunks
        chunks_cnt = 0
        for chunk in chunks:
            chunks_cnt += len(chunk)

        print(f"[INFO] Splitted {docs_cnt} documents into {chunks_cnt} chunks")
        
        return chunks
    
    def embed_chunks(self, chunks_list: List[Any]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks

        Args:
            List of texts to embed
        Returns: 
            numpy array of embeddings with shape
        """
        if not self.model_name:
            raise ValueError("Model not loaded yet!")
        
        # Create a list with page_content of a chunk
        texts = []
        for chunks in chunks_list:
            texts.extend([chunk.page_content for chunk in chunks])
 
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
if __name__ == "__main__":

    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()

    # docs -> List(Documents)
    chunks = emb_pipe.chunk_documents(docs)

    # chunks -> ndarray[Any shape]
    embeddings = emb_pipe.embed_chunks(chunks)

    # print(embeddings)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)

