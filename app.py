import streamlit as st
import logging
import time
from utils.file_handler import handle_uploaded_file
from utils.chat_response import generate_response
from utils.logger import setup_logger


setup_logger()


if "messages" not in st.session_state:
    st.session_state.messages = []  

def main():
    st.title("AI-Driven Document Chatbot (Transformer Model)")

    st.header("Upload a Document or Image")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "jpg", "png"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        text = handle_uploaded_file(uploaded_file, file_type)

        if text:
           
            st.text_area("Extracted Text", value=text, height=300, disabled=True)
            logging.info(f"Document uploaded and processed ({uploaded_file.name})")

            

           
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question based on the document:"):
                with st.chat_message("user"):
                    st.markdown(prompt)

                st.session_state.messages.append({"role": "user", "content": prompt})

                start_time = time.time()

                response, confidence, response_time, extracted_sentence = generate_response(prompt, text)

                end_time = time.time()
                response_time = end_time - start_time

                logging.info(f"Generated response: {response}")
                logging.info(f"Response time: {response_time:.4f} seconds")

                with st.chat_message("bot"):
                    st.markdown(response)
                    st.markdown(f"Extracted sentence from text: {extracted_sentence}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"Confidence Score: {confidence:.2f}")

                    with col2:
                        st.markdown(f"Response Time: {response_time:.4f} seconds")

                st.session_state.messages.append({"role": "bot", "content": response})

        else:
            logging.error("File could not be processed.")
            st.error("Could not process the file. Please upload a valid document.")
            
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
