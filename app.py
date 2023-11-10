from rag_function import rag
from rag_function import memory
import streamlit as st

st.title('Ugandan Laws and Regulations')
progress_text = "Operation in progress. Please wait."

# set initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there! Ask me something about the Ugandan Laws and Regulation"}
    ]

if "messages" in st.session_state.keys():
    # display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# get user input
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(progress_text):
            ai_response = rag(user_prompt)
            st.write(ai_response)
            
    new_ai_message = {"role": "user", "content": ai_response}
    st.session_state.messages.append(new_ai_message)

with st.sidebar:
    st.title ('Some Tips')
    st.write ('Provide as much context within your questions as possible for me to refer to the most relevant sources')
    st.write ('So far I only have a database of Petroleum and Tax laws')
    st.write ('To clear the memory of your chat history and start afresh, click the button below')
    if st.button("Clear Chat history"):
        print("Clearing message history")
        memory.clear()
        st.session_state.trace_link = None
        st.session_state.run_id = None
