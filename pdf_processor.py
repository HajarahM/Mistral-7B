import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

class DocumentProcessor:
    """
    A class for processing PDF documents from a directory, creating embeddings, and saving them in a single vector store.

    Args:
        input_dir (str): The input directory containing PDF documents.
        output_dir (str): The output directory to store embeddings.

    Methods:
        process_documents(): Process all PDF documents in the input directory, create embeddings, and save them in the output directory.
    """

    def __init__(self, input_dir, output_dir, embeddings):
        """
        Initializes the DocumentProcessor.

        Args:
            input_dir (str): The input directory containing PDF documents.
            output_dir (str): The output directory to store embeddings.
            embeddings: An embeddings model for creating embeddings.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.embeddings = embeddings
        self.embeddings_list = []

    def process_documents(self):
        """
        Process all PDF documents in the input directory, create embeddings, and save them in a single output directory.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.embeddings_list = []

        for filename in os.listdir(self.input_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.input_dir, filename)
                text = self.extract_text_from_pdf(pdf_path)
                embeddings = self.create_embeddings(text)
                self.embeddings_list.extend(embeddings)

        #self.save_embeddings(self.embeddings_list)

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: Extracted text from the PDF file.
        """
        loader = PyPDFDirectoryLoader(pdf_path)
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

    def save_embeddings(self, document_chunks):
        """
        Save embeddings in the output directory using FAISS.

        Args:
            document_chunks (list): List of document chunks with embeddings.
        """
        persist_directory = "vdb_legal_docs"
        vector_store = Chroma.from_documents(document_chunks, self.embeddings, persist_directory)
        vector_store.persist()
        
if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="mistral")

    input_dir = "data/"  # Replace with the path to your input directory
    output_dir = "vdb_legal_docs"  # Replace with the path to your output directory

    processor = DocumentProcessor(input_dir, output_dir, embeddings)
    processor.process_documents()