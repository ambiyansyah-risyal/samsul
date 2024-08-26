import streamlit as st
from rag import RAG

rag = RAG()

def reset_conversation():
    st.session_state.conversation = None
    st.session_state.messages = []

def show_ui():
    st.set_page_config(page_title="SAMSUL", page_icon=':robot_face:')
    st.title("Welcome to SAMSUL")
    st.header("Your Intelligent Companion for Every Conversation")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Message SAMSUL...")

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Processing"):
            response = rag.chain(prompt)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

            with st.chat_message('assistant'):
                st.markdown(response)

    st.button('Reset Chat', on_click=reset_conversation)

if __name__ == "__main__":
    show_ui()
