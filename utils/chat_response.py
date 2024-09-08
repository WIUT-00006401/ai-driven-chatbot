import logging
from transformers import pipeline
from utils.logger import setup_logger
from utils.text_processing import split_text_into_chunks
import nltk
import time
import os


setup_logger()


# nltk.download('punkt')

nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path + '/tokenizers/punkt'):
    nltk.download('punkt', download_dir=nltk_data_path)


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

    try:
        start_time = time.time()

        chunks = split_text_into_chunks(document_text)
        best_answer = ""
        best_score = 0
        best_chunk = ""

        for chunk in chunks:
            response = qa_pipeline(question=question, context=chunk)
            if response['score'] > best_score:
                best_answer = response['answer']
                best_score = response['score']
                best_chunk = chunk 


        best_sentence = find_sentence_with_answer(best_chunk, best_answer)

        if best_sentence:
            logging.info(f"Answer extracted from sentence: {best_sentence}")
        else:
            logging.info(f"Answer found but unable to extract specific sentence.")
            
        if best_score < 0.5:
            logging.info(f"Low confidence in answer: {best_answer}, Score: {best_score}")
            return "This information is not available in the document.", 0, 0, best_sentence

        end_time = time.time()
        response_time = end_time - start_time

        logging.info(f"Selected answer from document: {best_answer} (score: {best_score})")

        return best_answer, best_score, response_time, best_sentence
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return "Sorry, I couldn't process the question.", 0, 0, "N/A"
