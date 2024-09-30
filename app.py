import streamlit as st
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI for OpenAI API key input
st.title("Virtual Customer")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

    # Choose the LLM model
    model_choice = st.selectbox("Select an LLM model", ["gpt-3.5-turbo", "gpt-4o-mini"])

    # Initialize the LLM
    st.write("Initializing LLM model...")
    llm = OpenAI(model=model_choice)

    # Query interaction using LLM
    st.write("Chat with the LLM")
    user_input = st.text_input("Ask your question:")

    if user_input:
        # System message to set up the assistant's role as a virtual customer
        messages = [
            ChatMessage(
                role="system", 
                content="You are a virtual customer designed to help upskill and train bank employees. Your role is to provide realistic banking scenarios, respond as a customer would, and evaluate the employees based on their responses. Sometimes you are angry, short-tempered, and in a hurry."
            ),
            ChatMessage(role="user", content=user_input),
        ]
        
        # Get response from LLM
        response = llm.chat(messages)

        # Add user input and LLM response to conversation history
        st.session_state.conversation_history.append(f"User: {user_input}")
        st.session_state.conversation_history.append(f"Assistant: {response}")

        # Display the assistant's response
        st.write(f"LLM Response: {response}")

    # Display conversation history
    if st.session_state.conversation_history:
        st.write("## Conversation History")
        for entry in st.session_state.conversation_history:
            st.write(entry)

    # Download conversation history as a text file
    if st.session_state.conversation_history:
        conversation_text = "\n".join(st.session_state.conversation_history)
        st.download_button(
            label="Download Conversation History",
            data=conversation_text,
            file_name="conversation_history.txt",
            mime="text/plain",
        )

    # Embeddings section
    st.write("## Embeddings Section")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini", max_tokens=300)

    # Document loading and querying
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded PDF locally
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load documents and create an index
        documents = SimpleDirectoryReader("./").load_data()
        index = VectorStoreIndex.from_documents(documents)

        # Query engine
        query = st.text_input("Good morning! Welcome to Canara Bank. How can I assist you today?:")
        if query:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)

            # Add query and response to conversation history
            st.session_state.conversation_history.append(f"User: {query}")
            st.session_state.conversation_history.append(f"Document Response: {response}")

            # Display response
            st.write(f"Document Response: {response}")
else:
    st.warning("Please enter your OpenAI API key to continue.")
