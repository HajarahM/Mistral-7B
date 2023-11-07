import streamlit as st
import numpy as np

#LOG ALL CHAT MESSAGES INTO chathistory.txt
def writehistory(text):
    with open('chathistory.txt','a') as f:
        f.write(text)
        f.write('\n')
    f.close()

#Page Title
st.title('Ugandan Law and Regulations Resource')

# Add a slectbox to the sidebar:
add_selectbox = st.sidebar.selectbox('Select a domain', ('Oil and Gas','Power', 'Mining'))

with st.chat_message("assistant"):
    st.write("Hello 👋")

prompt = st.chat_input(placeholder="Type your message here", key=None, max_chars=None, disabled=False, on_submit=None, args=None, kwargs=None)

#use @st.cache_data for a function activity you want to recall
# use st.cache_resource for ML models and database connections


