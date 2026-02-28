import os
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()

class AskLLM:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", model_name: str = "deepseek/deepseek-r1-0528:free") -> None:
        """ 
            Contructor: Loads vectorstore and setup the LLM
            Args:
                persist_dir: str
                embedding_model: str
                model_name: str
        """
        self.model_name = model_name
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        api_key = os.getenv("GENAI_API_KEY")

        # Check whether the api_key got value
        if not api_key:
            print("[INFO] API key not found in environment variables")
            raise
        else:
            print("[INFO] API key is successfully loaded!")
        
        # Initialize the ChatOpenAI model with OpenRouter's details
        try:
            self.llm_model = ChatOpenAI(
                model=model_name,  # The specific model name from OpenRouter
                base_url="https://openrouter.ai/api/v1", # OpenRouter's API endpoint
                api_key=api_key
            )
        except Exception as e:
            print(f"[ERROR] Failed while initialize the llm model {model_name} : {e}")
        
        print(f"[INFO] LLM Model {model_name} loaded Successfully!")

    def ask(self, query: str, top_k: int = 5) -> str:
        """  
            Utilize VectorStore class to retrieve related raw chunks from loaded documents. Pass the context and query to LLM and gets response
            Args:
                query: str -> user' query
                top_k: int -> number of top matching raw chunks as context
            Return:
                response.content: str -> LLM's response
        """
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        prompt = f"""Answer the question only using the provided context.
                    If the answer is not present, say "I don't know". 

                    Context:
                    {context}

                    Question:
                    {query}
                    """
        try:
            response = self.llm_model.invoke([prompt])
            print(f"[INFO] Got response from LLM model : {self.model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to get response from LLM : {e}")
            
        return response.content

if __name__ == "__main__":
    llm = AskLLM()
    print(f"[INFO] LLM response : {llm.ask('What is money market?')}")





