import os
import requests
from bs4 import BeautifulSoup

class LegalFetcher:
    """
    A class to fetch legislation and regulations from the Uganda Legal Information Institute website,
    download all the PDF documents, and save them locally in their respective folders.

    Attributes:
        legislation_url (str): The URL for legislation documents.
        regulations_url (str): The URL for regulations documents.
        folder_path (str): The base path of the folder to store the downloaded PDF documents in.

    Methods:
        fetch_legal_docs(): Fetches all legislation and regulations from the website and downloads all the PDF documents.
    """

    def __init__(self, legislation_url: str, regulations_url: str, parliament_url:str, folder_path: str):
        """
        Initializes the LegalFetcher class with the given URLs and folder path.
        """
        self.legislation_url = legislation_url
        self.regulations_url = regulations_url
        self.parliament_url = parliament_url
        self.folder_path = folder_path

    def fetch_legal_docs(self) -> None:
        """
        Fetches all legislation and regulations from the website and downloads all the PDF documents.
        """
        # Create base folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        categories = {
            "legislation": self.legislation_url,
            "regulations": self.regulations_url,
            "parliament": self.parliament_url
        }

        for category, category_url in categories.items():
            category_folder_path = os.path.join(self.folder_path, category)
            if not os.path.exists(category_folder_path):
                os.makedirs(category_folder_path)

            # Fetch all legal document links
            response = requests.get(category_url)
            soup = BeautifulSoup(response.content, "html.parser")
            legal_doc_links = soup.select(".document-link a")

            # Download PDFs for each legal document link
            for link in legal_doc_links:
                pdf_link = link.get("href")
                pdf_name = pdf_link.split("/")[-1]
                pdf_path = os.path.join(category_folder_path, pdf_name)

                # Check if file already exists
                if os.path.exists(pdf_path):
                    print(f"{pdf_name} already exists. Skipping...")
                    continue

                # Download PDF
                response = requests.get(pdf_link)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

if __name__ == "__main__":
    legislation_url = "https://ulii.org/legislation"
    regulations_url = "https://www.pau.go.ug/regulations/"
    parliament_url = "https://www.parliament.go.ug"
    folder_path = "legal_docs"
    legal_fetcher = LegalFetcher(legislation_url, regulations_url, parliament_url, folder_path)
    legal_fetcher.fetch_legal_docs()