import logging
from transformers import pipeline
from utils.logger import setup_logger
from utils.text_processing import split_text_into_chunks, handle_future_or_speculative_questions
import nltk
import time

# Set up the logger
setup_logger()

# Download the required NLTK data for sentence tokenization
nltk.download('punkt')

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def find_sentence_with_answer(chunk, answer):
    """Find the sentence in the chunk that contains the answer."""
    sentences = nltk.sent_tokenize(chunk)
    for sentence in sentences:
        if answer in sentence:
            return sentence
    return None

def generate_response(question, document_text):
    logging.info(f"Received question: {question}")
    
    # Check for speculative questions
    # speculative_response = handle_future_or_speculative_questions(question)
    # if speculative_response:
    #     return speculative_response, 0, 0

    try:
        # Start time for measuring response time
        start_time = time.time()

        # Split the text into chunks for better processing
        chunks = split_text_into_chunks(document_text)
        best_answer = ""
        best_score = 0
        best_chunk = ""

        # Process each chunk and get the best answer
        for chunk in chunks:
            response = qa_pipeline(question=question, context=chunk)
            if response['score'] > best_score:
                best_answer = response['answer']
                best_score = response['score']
                best_chunk = chunk  # Store the chunk from which the answer was selected


        # Find the sentence in the best chunk that contains the answer
        best_sentence = find_sentence_with_answer(best_chunk, best_answer)

        # Log the selected sentence where the answer was found
        if best_sentence:
            logging.info(f"Answer extracted from sentence: {best_sentence}")
        else:
            logging.info(f"Answer found but unable to extract specific sentence.")
            
        # Implement a confidence threshold
        if best_score < 0.5:
            logging.info(f"Low confidence in answer: {best_answer}, Score: {best_score}")
            # return "This information is not available in the document.", best_score, 0
            return "This information is not available in the document.", 0, 0, best_sentence

        # End time for measuring response time
        end_time = time.time()
        response_time = end_time - start_time

        logging.info(f"Selected answer from document: {best_answer} (score: {best_score})")

        return best_answer, best_score, response_time, best_sentence
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return "Sorry, I couldn't process the question.", 0, 0, "N/A"
