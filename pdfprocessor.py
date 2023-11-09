import os
from llmsherpa.readers import LayoutPDFReader
from langchain.vectorstores import faiss
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import PyPDFium2Loader
import ocrmypdf
from PIL import Image
import pytesseract

class DocumentProcessor:
    """
    A class for processing PDF documents, creating embeddings, and saving them in a FAISS vector store.

    Args:
        raw_files_dir (str): The source directory containing PDF documents.
        vector_store_dir (str): The directory to save the FAISS vector store.
        embeddings: An embeddings model for creating embeddings.

    Methods:
        process_documents(): Process PDF documents, create embeddings, and save them in a FAISS vector store.
    """

    def __init__(self, raw_files_dir, vector_store_dir, embeddings):
        """
        Initializes the DocumentProcessor.

        Args:
            raw_files_dir (str): The source directory containing PDF documents.
            vector_store_dir (str): The directory to save the FAISS vector store.
            embeddings: An embeddings model for creating embeddings.
        """
        self.raw_files_dir = raw_files_dir
        self.vector_store_dir = vector_store_dir
        self.embeddings = embeddings

    def process_documents(self):
        """
        Process PDF documents, create embeddings, and save them in a FAISS vector store.
        """
        if not os.path.exists(self.vector_store_dir):
            os.makedirs(self.vector_store_dir)

        for root, dirs, files in os.walk(self.raw_files_dir):
            for filename in files:
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(root, filename)
                    text = self.convert_pdf_to_text(pdf_path)
                    embeddings = self.create_embeddings(text)
                    self.save_embeddings(embeddings, filename)
                    text_embedding_pairs = zip(text, embeddings)
                    text_embedding_pairs_list = list(text_embedding_pairs)
                    faiss = faiss.from_embeddings(text_embedding_pairs_list, embeddings)
                    faiss.save_local
        
    def convert_pdf_to_text(self, pdf_path):
        """
        Convert a PDF to text using OCR.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: Extracted text from the PDF file.
        """
        output_path = os.path.join(self.vector_store_dir, os.path.basename(pdf_path))
        loader = ocrmypdf(output_path)
        data = loader.load()
        text = self.extract_text_from_pdf(data)
        return text

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: Extracted text from the PDF file.
        """
        loader = PyPDFium2Loader(pdf_path)
        text = loader.load()
        return text

    def create_embeddings(self, text):
        """
        Create embeddings from text using the specified embeddings model.

        Args:
            text (str): Text to create embeddings from.

        Returns:
            list: List of document chunks with embeddings.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20, separators=['\n\n', '\n', '.'])
        document_chunks = text_splitter.split_documents(text)
        return document_chunks

    def save_embeddings(self, document_chunks, filename):
        """
        Save embeddings in a separate FAISS index.

        Args:
            document_chunks (list): List of document chunks with embeddings.
            filename (str): The filename to use for the index.
        """
        index_path = os.path.join(self.vector_store_dir, filename)
        vector_store = FAISS.from_documents(text_chunks, embedding=embedding_model)

        for doc_chunk in document_chunks:    
            folder_path="FAISS_vector_store"
            if os.path.exists(folder_path):
                faiss_index=FAISS.load_local(folder_path, embeddings)
                faiss_index.merge_from(vector_store)
                faiss_index.save_local(folder_path)
            else:
                vector_store.save_local(folder_path)
                    # Save the persistent database
                    index.persist(index_path)

        faiss = FAISS.from_embeddings(text_embedding_pairs_list, embeddings)

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="mistral")

    raw_files_dir = "laws/Petroleum"  # Replace with the path to your source directory
    vector_store_dir = "FAISS_vector_store"  # The directory to save the FAISS vector store

    processor = DocumentProcessor(raw_files_dir, vector_store_dir, embeddings)
    processor.process_documents()