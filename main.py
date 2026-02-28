from src.data_loader import load_all_documents


if __name__ == "__main__":
    all_docs = load_all_documents("data")
    print(all_docs)
