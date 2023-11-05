import os
from faiss import FlannIndex, IndexFlann
import pyocr
from PIL import Image
from mistral import Mistral
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LawsClassifier:
    def __init__(self):
        self.model = Mistral()
        self.tokenizer = AutoTokenizer.from_pretrained("mistral/bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.faiss_index = None

    def load_documents(self, folder):
        self.load_pdf_to_text(folder)
        self.load_image_to_text(folder)
        self.chunk_documents(folder)
        self.create_faiss_index(folder)

    def load_pdf_to_text(self, folder):
        pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            filepath = os.path.join(folder, pdf_file)
            with open(filepath, "rb") as file:
                text = pyocr.tesseract_cmd("eng", pageSegMode=3, lang="eng", ocrEngineMode=pyocr.OCREngineMode.LATEST)
                if len(text) > 0:
                    self.model.tokenize(text)

    def load_image_to_text(self, folder):
        image_files = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".jpeg")]
        for image_file in image_files:
            filepath = os.path.join(folder, image_file)
            img = Image.open(filepath)
            self.model.tokenize(img)

    def chunk_documents(self, folder):
        documents = []
        for sub_folder in os.listdir(folder):
            sub_folder_path = os.path.join(folder, sub_folder)
            if os.path.isdir(sub_folder_path):
                documents += self.chunk_documents(sub_folder_path)
        self.model.tokenize(" ".join(documents))

    def create_faiss_index(self, folder):
        if self.faiss_index is not None:
            self.faiss_index.delete()

        index = FlannIndex(K=1024)
        index_flann = IndexFlann(index, flann_database_file="faiss.db")
        self.faiss_index = index_flann

        for sub_folder in os.listdir(folder):
            sub_folder_path = os.path.join(folder, sub_folder)
            if os.path.isdir(sub_folder_path):
                sub_folder_index = FlannIndex(K=1024)
                self.faiss_index.add_subindex(sub_folder_index)
                self.faiss_index.update_subindex(sub_folder_index)

    def get_embeddings(self, folder):
        embeddings = self.model.encode(self.tokenizer.encode(folder))
        return embeddings