import os
#import fitz  # PyMuPDF
from llmsherpa.readers import LayoutPDFReader
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

class DocumentProcessor:
    def __init__(self, input_dir, output_dir, pdf_reader, embeddings):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pdf_reader = pdf_reader
        self.embeddings = embeddings

    def process_documents(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for category in os.listdir(self.input_dir):
            category_path = os.path.join(self.input_dir, category)
            if not os.path.isdir(category_path):
                continue

            output_category_path = os.path.join(self.output_dir, category)
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            for filename in os.listdir(category_path):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(category_path, filename)
                    text = self.extract_text_from_pdf(pdf_path)
                    embeddings = self.create_embeddings(text)
                    self.save_embeddings(embeddings, output_category_path, filename)

    def extract_text_from_pdf(self, pdf_path):
        doc = self.pdf_reader.read_pdf(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def create_embeddings(self, text):
        text_splitter = RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=10, separators=['\n\n', '\n', '.'])
        document_chunks = text_splitter.split_documents(text)
        return document_chunks

    def save_embeddings(self, document_chunks, output_dir, filename):
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.emb")
        #Initiate a Chromadb instance
        vectordb = Chroma.from_documents(document_chunks, self.embeddings, output_path)
        #save persist database
        vectordb.persist()
        vectordb = None

if __name__ == "__main__":
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    embeddings = OllamaEmbeddings(model="mistral")

    input_dir = "legal_docs"  # Create if not there with code inside the class
    output_dir = "vdb_legal_docs" # Create if not there with code inside the class

    processor = DocumentProcessor(input_dir, output_dir, pdf_reader, embeddings)
    processor.process_documents()
