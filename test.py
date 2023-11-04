# Import libraries
import os
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# SET LLM AND EMBEDDINGS MODEL
#Ollama Embeddings
embeddings_open = OllamaEmbeddings(model="mistral")

#optional embeddeing model
embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-zh-v1.5')

#llm
llm = Ollama(
    model = "mistral",
    # or use Llama2
    # model = "Llama2",
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    )

#LOAD DATA FROM DIRECTORY

#Print number of txt files in specified directory (data)
loader = DirectoryLoader('./data', glob="./*.txt")

#load pdfs from directory and print number of pdfs
loader = DirectoryLoader('./data', glob="./*.pdf")

doc = loader.load()

len(doc)

#SPLIT DOCUMENTS INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=10, separators=['\n\n', '\n', '.'])
document_chunks = text_splitter.split_documents(doc)

len(document_chunks)

# Initiate a chromadb instance
chroma_db = Chroma.from_documents(document_chunks, embedding_model)
retriever = chroma_db.as_retriever()

#PDFs from directory
persist_directory = 'vdb_arbitration'

#Initiate a Chromadb instance
vectordb = Chroma.from_documents(
    documents=document_chunks,
    embedding=embeddings_open,
    persist_directory=persist_directory
)

#save persist database
vectordb.persist()
vectordb = None

#Documents from directory
persist_directory = 'vdb_arbitration'

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings_open
)

retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("what is this document about")
#CREATE QUESTION ANSWERING (QA) CHAIN

#Option 1
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#Option 2
# # Prompt Template
# qa_template = """ <s>[INST] You are a legal assistan.
# Use the following context to Answer the question below briefly:
# {context: You are a library of ugandan law and you answer questions regarding what the law says about particular topics specifically from the documents}
# {question}[/INST]<s>
# # """
# #Create a prompt instance
# QA_PROMPT = PromptTemplate.from_template(qa_template)

# #Custom QA Chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, 
#     retriever = retriever, 
#     chain_type_kwargs={"prompt":QA_PROMPT}
#     )
query = "What is this document about?"
llm_response = qa_chain(query)