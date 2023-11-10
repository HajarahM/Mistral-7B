from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

embedding_model = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')

#load pdf files
documents = []
processed_directories=0
for dir in os.listdir("data"):
    try: 
        dir_path = './data/'+dir
        loader = PyPDFDirectoryLoader(dir_path)
        documents.extend(loader.load())
        processed_directories+=1
    except:
        print("issue with ", dir)
        pass
print("processed ",processed_directories," directories")

#Split the Extracted Data into Text Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n', '.'])

text_chunks = text_splitter.split_documents(documents)

len(text_chunks)

#Create Embeddings for each of the Text Chunk
import os
vector_store = FAISS.from_documents(text_chunks, embedding=embedding_model)
print('saving embeddings to vector_store')
folder_path="FAISS_vector_store"
if os.path.exists(folder_path):
    faiss_index=FAISS.load_local(folder_path, embedding_model)
    faiss_index.merge_from(vector_store)
    faiss_index.save_local(folder_path)
else:
    vector_store.save_local(folder_path)

