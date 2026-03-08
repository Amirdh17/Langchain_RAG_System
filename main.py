from src.data_loader import load_all_documents, move_all_files
import src.vectorstore
from src.vectorstore import FaissVectorStore
import src.search
from src.search import AskLLM


if __name__ == "__main__":
    """ 
        This is a main function for this project. This function ask user to upload the required documents
        into the folder "Data". If you already uploaded your files, then you can skip the document loading part by giving input as 'No' and system will automatically loads any persisted data from vector store. Once it got confirmation from user for uploading the user's files at mentioned
        folder, it will load all files from the folder 'Data' and store them into the vector store which will persist permanently. Then it will move loaded files into 'archive' folder.
        User needs to give his/her question when system ask for it and LLM provide the answer based on the content of 
        the documents loaded into vector store.
    """
    print("[SYSTEM] Hi:) there! I'm Q & A system. I would like to answer your question by using your docs / files .\nSupported file types are pdf, txt, csv, excel, word & json")

    # Get user's response
    user_response = ''
    while (user_response == '' or user_response.lower() not in ('no', 'yes', 'n', 'y')):
        user_response = input("[INPUT] Would you like to upload your docs? if yes, please upload your docs at 'Data' folder and give input as 'Yes'. otherwise give input as 'No' [Yes/No] :")
        if (user_response.lower() not in ('no', 'yes', 'n', 'y')):
            print("[ERROR] Please answer only by using yes/y or no/n" )

    # Loads files present in Data folder
    if user_response.lower() in ('yes', 'y'):
        all_docs = load_all_documents("data")

        # Moves loaded files into Archive folder
        move_all_files("data", "archive")

    # Initialize vector store
    store = FaissVectorStore()

    if not store.load():
        store.build_from_documents(all_docs)
    
    # Initialize the LLM 
    llm = AskLLM()

    query = input("[INPUT] Hi, Please ask any question related to the documents :")

    # Passing query to the llm
    response = llm.ask(query, top_k=5)

    print(f"LLM response: {response}")

    

    


