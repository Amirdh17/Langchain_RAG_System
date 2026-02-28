import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from pathlib import Path
from typing import List, Any
import os
import shutil



def load_all_documents(data_directory: str) -> List[Any]:
    """  
        This function read different type of file from the given directory and store them into a List.
        Supported files: pdf, txt, csv, excel, word & json
        Args:
            data_directory: str
        Return:
            document: List[str]
    """
    all_documents = []
    data_dir = Path(data_directory)
    print(f"[DEBUG] Data Path: {data_dir}")

    # Load all pdf files into the document
    pdf_path_list = list(data_dir.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_path_list)} pdf files: {[str(f) for f in pdf_path_list]}")

    for pdf_path in pdf_path_list:
        print(f"[DEBUG] Processing pdf file: {pdf_path}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            # add source information to metadata
            for doc in docs:
                doc.metadata['source_file'] = pdf_path.name
                doc.metadata['file_type']  = 'pdf'

            all_documents.append(docs)
            print(f"[DEBUG] Loaded {len(docs)} pdf files from {pdf_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load pdf file {pdf_path} : {e}")
    
    # load all text files
    txt_path_list = list(data_dir.glob("**/*.txt"))
    print(f"[DEBUG] Found {len(txt_path_list)} text files: {[str(f) for f in txt_path_list]}")

    for txt_path in txt_path_list:
        print(f"[DEBUG] Processing txt file: {txt_path}")

        try:
            loader = TextLoader(str(txt_path))
            docs = loader.load()

            # add source information to metadata
            for doc in docs:
                doc.metadata['source_file'] = txt_path.name
                doc.metadata['file_type']  = 'txt'

            all_documents.append(docs)
            print(f"[DEBUG] Loaded {len(docs)} text docs from {txt_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load text file {txt_path} : {e}")
    
    # Load .csv files
    csv_path_list = list(data_dir.glob("**/*.csv"))
    print(f"[DEBUG] Found {len(csv_path_list)} csv files: {[str(f) for f in csv_path_list]}")

    for csv_path in csv_path_list:
        print(f"[DEBUG] Processing csv file: {csv_path}")

        try:
            loader = CSVLoader(str(csv_path))
            docs = loader.load()

            # add source information to metadata
            for doc in docs:
                doc.metadata['source_file'] = csv_path.name
                doc.metadata['file_type']  = 'csv'

            all_documents.append(docs)
            print(f"[DEBUG] Loaded {len(docs)} csv docs from {csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load csv file {csv_path} : {e}")
    
    # Load Excel (.xlsx) files
    xlsx_path_list = list(data_dir.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_path_list)} Excel files: {[str(f) for f in xlsx_path_list]}")

    for xlsx_path in xlsx_path_list:
        print(f"[DEBUG] Loading Excel: {xlsx_path}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_path))
            docs = loader.load()

            # add source information to metadata
            for doc in docs:
                doc.metadata['source_file'] = xlsx_path.name
                doc.metadata['file_type']  = 'xlsx'

            print(f"[DEBUG] Loaded {len(docs)} Excel docs from {xlsx_path}")
            all_documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_path}: {e}")
    
    # Load word (.docx) files
    docx_path_list = list(data_dir.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_path_list)} Word files: {[str(f) for f in docx_path_list]}")
    for docx_path in docx_path_list:
        print(f"[DEBUG] Loading Word: {docx_path}")
        try:
            loader = Docx2txtLoader(str(docx_path))
            docs = loader.load()

            # add source information to metadata
            for doc in docs:
                doc.metadata['source_file'] = docx_path.name
                doc.metadata['file_type']  = 'docx'

            print(f"[DEBUG] Loaded {len(docs)} Word docs from {docx_path}")
            all_documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_path}: {e}")

    # Load JSON files
    json_path_list = list(data_dir.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_path_list)} JSON files: {[str(f) for f in json_path_list]}")
    for json_path in json_path_list:
        print(f"[DEBUG] Loading JSON: {json_path}")
        try:
            loader = JSONLoader(str(json_path))
            docs = loader.load()

            # add source information to metadata
            for doc in docs:
                doc.metadata['source_file'] = json_path.name
                doc.metadata['file_type']  = 'json'
            
            print(f"[DEBUG] Loaded {len(docs)} JSON docs from {json_path}")
            all_documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_path}: {e}")
        
    print(f"[DEBUG] Total loaded documents; {len(all_documents)}")
    
    return all_documents

def move_all_files(source_folder: str, destination_folder: str) -> None:
    """
    Copies all files from source (including subfolders)
    to destination and deletes the original files.
    Args:
        source_folder: str
        destination_folder: str
    """

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_file_path = os.path.join(root, file)   

            # Preserve folder structure
            relative_path = os.path.relpath(root, source_folder)
            destination_dir = os.path.join(destination_folder, relative_path)

            os.makedirs(destination_dir, exist_ok=True)

            destination_file_path = os.path.join(destination_dir, file)

            try:
                # Copy file
                shutil.copy2(source_file_path, destination_file_path)

                # Delete original file after successful copy
                os.remove(source_file_path)

                print(f"[INFO] Moved: {source_file_path}")

            except Exception as e:
                print(f"[ERROR] failed while moving {source_file_path}: {e}")

    print("[INFO] Loaded files are moved to archive folder successfully!")

if __name__ == "__main__":
    loaded_docs = load_all_documents("data")

    if len(loaded_docs) > 0:
        print(f"[DEBUG] Successfully! Loaded {len(loaded_docs)} documents")
    else:
        print(f"[DEBUG] No files found at the given path: {Path('data')}")
    
    # Once loaded, moving files to archive
    # move_all_files("../data", "../archive")