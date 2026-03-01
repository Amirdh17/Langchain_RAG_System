from src.data_loader import load_all_documents, move_all_files
import src.vectorstore
from src.vectorstore import FaissVectorStore
import src.search
from src.search import AskLLM


if __name__ == "__main__":
    # Loads files present in Data folder
    all_docs = load_all_documents("data")
    
    # Moves loaded files into Archive folder
    move_all_files("../data", "../archive")

    # Initialize vector store
    store = FaissVectorStore()

    if not store.load():
        store.build_from_documents(all_docs)
    
    # Initialize the LLM 
    llm = AskLLM()

    query = input("Hi, Please ask any question related to the documents :")

    # Passing query to the llm
    response = llm.ask(query, top_k=5)

    print(f"LLM response: {response}")

    

    


