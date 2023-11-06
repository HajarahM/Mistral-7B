import os
from llmsherpa.readers import LayoutPDFReader
from langchain.vectorstores import FAISS
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
        index = FAISS()
        for doc_chunk in document_chunks:
            index.insert([doc_chunk])
        # Save the persistent database
        index.persist(index_path)

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="mistral")

    raw_files_dir = "laws/Petroleum"  # Replace with the path to your source directory
    vector_store_dir = "FAISS_vector_store"  # The directory to save the FAISS vector store

    processor = DocumentProcessor(raw_files_dir, vector_store_dir, embeddings)
    processor.process_documents()