import streamlit as st
import logging
import time
from utils.file_handler import handle_uploaded_file
from utils.chat_response import generate_response
from utils.logger import setup_logger

# Set up the logger
setup_logger()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # To store chat history

def main():
    st.title("AI-Driven Document Chatbot (Transformer Model)")

    st.header("Upload a Document or Image")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "jpg", "png"])

    if uploaded_file is not None:
        # Detect file type and handle accordingly
        file_type = uploaded_file.name.split('.')[-1].lower()
        text = handle_uploaded_file(uploaded_file, file_type)

        if text:
            # Display extracted document text for reference
            st.text_area("Extracted Text", value=text, height=300, disabled=True)
            logging.info(f"Document uploaded and processed ({uploaded_file.name})")

            # Add summarization functionality
            # if st.button("Summarize Document"):
            #     try:
            #         summary = summarize_text(text)
            #         st.subheader("Summarized Text")
            #         st.write(summary)
            #         logging.info(f"Document summarized successfully.")
            #     except Exception as e:
            #         st.error(f"Error summarizing document: {e}")
            #         logging.error(f"Error summarizing document: {e}")

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input in the chat-like interface
            if prompt := st.chat_input("Ask a question based on the document:"):
                # Display the user's message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Add the user's message to the chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Measure response time
                start_time = time.time()

                # Generate a response for the given question
                response, confidence, response_time, extracted_sentence = generate_response(prompt, text)

                # Measure the end time and calculate response time
                end_time = time.time()
                response_time = end_time - start_time

                logging.info(f"Generated response: {response}")
                logging.info(f"Response time: {response_time:.4f} seconds")

                # Display the bot's response in the chat interface
                with st.chat_message("bot"):
                    st.markdown(response)
                    st.markdown(f"Extracted sentence from text: {extracted_sentence}")

                    # Create two columns for displaying confidence and response time side by side
                    col1, col2 = st.columns(2)

                    # Display confidence score in the first column
                    with col1:
                        st.markdown(f"Confidence Score: {confidence:.2f}")

                    # Display response time in the second column
                    with col2:
                        st.markdown(f"Response Time: {response_time:.4f} seconds")

                # Add the bot's response to the chat history
                st.session_state.messages.append({"role": "bot", "content": response})

        else:
            logging.error("File could not be processed.")
            st.error("Could not process the file. Please upload a valid document.")
            
        # Provide a download button for the log file
        with open("chatbot_log.log", "r") as log_file:
            log_data = log_file.read()

        st.download_button(
            label="Download Log File",
            data=log_data,
            file_name="chatbot_log.log",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
