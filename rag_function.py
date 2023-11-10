from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

llm = Ollama(
    base_url="http://localhost:11434",
    model = "dolphin2.2-mistral:7b-q6_K",
    verbose=True,
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    )

embedding_model = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')

# create prompt
QA_prompt = PromptTemplate(
    template="""You are a friendly lawyer. Be thorough in your search within the context of the documents in the database and detailed in your response. 
    After end of each sentence move to the next line. At the end of the response name the specific source document.
    Do not add context to initial greeting. When someone says hello, respond with only a greeting.
    Use the following pieces of context to answer the user question. 
chat_history: {chat_history}
Context: {text}
Question: {question}
Answer:""",
    input_variables=["text", "question", "chat_history"]
) 

# create memory
memory = ConversationBufferMemory(
    return_messages=True, 
    memory_key="chat_history")

# create converstational retriever chain
def retrieval_qa_chain(llm, vectorstore):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(
            search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
        chain_type="refine",
    )
    return qa_chain

def qa_bot():
    db_path = "FAISS_vector_store/"
    vectorstore = FAISS.load_local(db_path,embedding_model)
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

def rag(question: str) -> str:
    # call QA chain
    qa = qa_bot()
    response = qa.run({"question": question})

    return response