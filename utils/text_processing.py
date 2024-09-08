# utils/text_processing.py

def split_text_into_chunks(text, chunk_size=300):
    """Split the document into smaller chunks to help with processing."""
    sentences = text.split('. ')
    chunks = ['. '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

def handle_future_or_speculative_questions(question):
    """Detect and handle speculative or future-based questions."""
    if "will" in question.lower() or "in the future" in question.lower():
        return "The document does not provide information about future predictions."
    return None
