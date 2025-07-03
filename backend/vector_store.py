from langchain_community.vectorstores import Chroma
from config import embeddings
from document_loader import load_pdfs_from_s3, load_excel_from_s3

# Global vector store and retriever
db = None
retriever = None


def initialize_vector_store():
    """Initialize or reinitialize the vector store from both PDF documents and Excel QA pairs"""
    global db, retriever
    
    # Load PDFs from S3
    pdf_documents = load_pdfs_from_s3()
    
    # Load Excel QA data as documents
    excel_documents = load_excel_from_s3()
    
    # Combine all documents
    all_documents = pdf_documents + excel_documents
    
    if all_documents:
        # Initialize Chroma with the combined documents
        db = Chroma.from_documents(all_documents, embeddings, persist_directory="./chroma_db")
        
        # Configure retriever with a proper configuration
        retriever = db.as_retriever(
            search_kwargs={
                "k": 5  # Return top 5 most relevant documents
            }
        )
        print(f"[INFO] Chroma vector store initialized with {len(all_documents)} documents "
              f"({len(pdf_documents)} PDF chunks and {len(excel_documents)} Excel QA pairs).")
        return True
    else:
        print("[WARN] No PDF documents or Excel QA pairs available to create vector store")
        return False


def get_retriever():
    """Get the current retriever instance"""
    return retriever


def is_vector_store_initialized():
    """Check if the vector store is initialized"""
    return retriever is not None