import os
import uuid
import boto3
import pandas as pd
from io import BytesIO
from typing import List
from botocore.exceptions import ClientError
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PDF_DIR, EXCEL_DIR, AWS_S3_BUCKET_NAME
from utils import extract_pdf_metadata


def load_pdfs_from_s3() -> List[Document]:
    """
    Load the pdf files from s3, where in the bucket, "/pdf-files" folder the PDFs are stored,
    fetch those files using boto3 and convert them to LangChain Document objects
    that can be used for the vector store db.
    
    Returns:
        List[Document]: List of Document objects created from pdf files.
    """
    try:
        # Initialize boto3 S3 client
        s3 = boto3.client('s3')
        
        # Get the bucket name from environment variable
        bucket_name = AWS_S3_BUCKET_NAME
        
        if not bucket_name:
            print("[ERROR] AWS_S3_BUCKET_NAME environment variable not set")
            return []
        
        # List objects in the PDF_DIR prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{PDF_DIR}/")
        
        if 'Contents' not in response:
            print(f"[INFO] No PDF files found in s3://{bucket_name}/{PDF_DIR}/")
            return []
        
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Process each PDF file
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directory objects or non-PDF files
            if not key.lower().endswith('.pdf'):
                continue
                
            print(f"\n[CONSOLE] Processing PDF from S3: {key}")
            
            try:
                # Create a temporary file
                temp_file_path = f"/tmp/{uuid.uuid4()}.pdf"
                
                # Download the file from S3
                s3.download_file(bucket_name, key, temp_file_path)
                print(f"[CONSOLE] Downloaded PDF to temporary path: {temp_file_path}")
                
                # Extract PDF metadata using PyPDF2 before loading with PyPDFLoader
                print(f"[CONSOLE] Extracting metadata from {temp_file_path}")
                pdf_metadata = extract_pdf_metadata(temp_file_path)
                
                # Check specifically for the Description field
                source_link = None
                # Check all possible Description key variations
                description_keys = ['Description', '/Description', 'description']
                for desc_key in description_keys:
                    if desc_key in pdf_metadata and pdf_metadata[desc_key]:
                        source_link = pdf_metadata[desc_key]
                        print(f"[CONSOLE] Found source link using key '{desc_key}': {source_link}")
                        break
                
                if not source_link:
                    print(f"[CONSOLE] No Description/source link found in metadata")
                
                loader = PyPDFLoader(temp_file_path)
                pdf_docs = loader.load()
                print(f"[CONSOLE] Loaded {len(pdf_docs)} pages from PDF")
                
                # Split the documents
                split_docs = text_splitter.split_documents(pdf_docs)
                print(f"[CONSOLE] Split into {len(split_docs)} document chunks")
                
                # Add metadata to identify the source
                for i, doc in enumerate(split_docs):
                    doc.metadata["source"] = key
                    doc.metadata["document_type"] = "pdf"
                    
                    # Add the source link from the PDF metadata if available
                    if source_link:
                        # Store the link in multiple fields to ensure it's found later
                        doc.metadata["source_link"] = source_link
                        doc.metadata["Description"] = source_link
                        doc.metadata["/Description"] = source_link
                        doc.metadata["description"] = source_link
                        print(f"[CONSOLE] Added source link to chunk {i} metadata")
                    
                    # Add all metadata fields for completeness
                    for meta_key, meta_value in pdf_metadata.items():
                        doc.metadata[meta_key] = str(meta_value)
                    
                    # Debug the first chunk's metadata
                    if i == 0:
                        print(f"[CONSOLE] Sample document chunk metadata: {doc.metadata}")
                
                documents.extend(split_docs)
                
                # Clean up temp file
                os.remove(temp_file_path)
                print(f"[CONSOLE] Removed temporary file {temp_file_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process PDF {key}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        print(f"[INFO] Successfully loaded {len(documents)} document chunks from {len(response['Contents'])} PDF files in S3")
        return documents
        
    except ClientError as e:
        print(f"[ERROR] AWS S3 client error: {str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error in load_pdfs_from_s3: {str(e)}")
        return []


def load_excel_from_s3() -> List[Document]:
    """
    Load question-answer pairs from Excel files stored in S3 (stored in the '/excel_files' directory
    of s3 bucket) directly in memory using boto3 and convert them to LangChain Document objects
    for vector store usage.
    
    Returns:
        List[Document]: List of Document objects created from Excel sheet QA pairs.
    """
    try:
        # Initialize boto3 S3 client
        s3 = boto3.client('s3')
        
        # Get the bucket name from environment variable
        bucket_name = AWS_S3_BUCKET_NAME
        
        if not bucket_name:
            print("[ERROR] AWS_S3_BUCKET_NAME environment variable not set")
            return []
        
        # List objects in the EXCEL_DIR prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{EXCEL_DIR}/")
        
        if 'Contents' not in response:
            print(f"[INFO] No Excel files found in s3://{bucket_name}/{EXCEL_DIR}/")
            return []
        
        documents = []
        
        # Process each Excel file
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directory objects or non-Excel files
            if not (key.lower().endswith('.xlsx') or key.lower().endswith('.xls')):
                continue
                
            print(f"[INFO] Processing Excel file from S3: {key}")
            
            try:
                # Get the file from S3 into memory
                response = s3.get_object(Bucket=bucket_name, Key=key)
                excel_data = BytesIO(response['Body'].read())
                
                # Read Excel file with pandas
                df = pd.read_excel(excel_data)
                
                # Ensure required columns exist
                required_columns = ['question', 'answer']
                if not all(col in df.columns for col in required_columns):
                    print(f"[WARN] Excel file {key} missing required columns (question, answer). Skipping.")
                    continue
                
                # Process each row in the Excel file
                for idx, row in df.iterrows():
                    question = str(row['question']).strip()
                    answer = str(row['answer']).strip()
                    
                    # Skip rows with empty questions or answers
                    if not question or not answer or question.lower() == 'nan' or answer.lower() == 'nan':
                        continue
                    
                    # Create a document from this QA pair
                    # Use the question as the page_content to match against user queries
                    doc = Document(
                        page_content=question,
                        metadata={
                            "source": key,
                            "document_type": "excel",
                            "question": question,
                            "answer": answer,
                            "row_index": idx
                        }
                    )
                    documents.append(doc)
            
            except Exception as e:
                print(f"[ERROR] Failed to process Excel file {key}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        print(f"[INFO] Successfully loaded {len(documents)} QA pairs from Excel files in S3")
        return documents
        
    except ClientError as e:
        print(f"[ERROR] AWS S3 client error: {str(e)}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error in load_excel_from_s3: {str(e)}")
        return []