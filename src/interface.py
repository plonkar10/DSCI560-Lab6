import streamlit as st
from htmlTemplates import bot_template, user_template, css

# Function to display the chat history
def display_chat():
    if "chat_history" in st.session_state:
        pairs = [st.session_state.chat_history[i : i + 2] for i in range(0, len(st.session_state.chat_history), 2)]
        reversed_pairs = reversed(pairs)

        # Iterate over reversed pairs to maintain order of conversation
        for pair in reversed_pairs:
            st.write(user_template.replace("{{MSG}}", pair[0].content), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", pair[1].content), unsafe_allow_html=True)

# Main function to render the interface
def build_interface():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PDFs :robot_face:")

    # Text input for user questions
    user_question = st.text_input("Ask questions about your documents:")

    if user_question and st.session_state.conversation:
        # Handle user input by passing it to the backend function
        handle_userinput(user_question)
        display_chat()
    elif user_question:
        st.warning("Please upload PDFs and click on 'Process' first.")

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF.")

# Call the function to build the interface
if __name__ == "__main__":
    build_interface()
