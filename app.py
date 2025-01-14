import gradio as gr
import numpy as np
import faiss
from mistralai import Mistral
import PyPDF2
import time
from tqdm import tqdm

# Initialize Mistral client
api_key = "API Key"
client = Mistral(api_key=api_key)

def extract_text_from_file(file_obj):
    """Extract text from either PDF or text file"""
    file_name = file_obj.name.lower()
    
    if file_name.endswith('.pdf'):
        # Handle PDF file
        pdf_reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif file_name.endswith('.txt'):
        # Handle text file
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Please upload either a PDF or text file.")

def get_text_embedding(input_text, max_retries=3, delay=1):
    """Compute text embeddings with retry logic"""
    for attempt in range(max_retries):
        try:
            embeddings_batch_response = client.embeddings.create(
                model="mistral-embed",
                inputs=input_text
            )
            return embeddings_batch_response.data[0].embedding
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = delay * (attempt + 1)  # Exponential backoff
                time.sleep(wait_time)
                continue
            raise e

def process_chunks_with_rate_limit(chunks, batch_size=5):
    """Process chunks with rate limiting and retries"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        batch_embeddings = []
        
        for chunk in batch:
            try:
                embedding = get_text_embedding(chunk)
                batch_embeddings.append(embedding)
                time.sleep(0.5)  # Add delay between requests
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                time.sleep(2)  # Longer delay on error
                try:
                    embedding = get_text_embedding(chunk)
                    batch_embeddings.append(embedding)
                except:
                    embedding = np.zeros(1024)
                    batch_embeddings.append(embedding)
        
        all_embeddings.extend(batch_embeddings)
        time.sleep(1)  # Delay between batches
    
    return np.array(all_embeddings)

def create_chunks(text, chunk_size=5000, overlap=200):
    """Create overlapping chunks from text"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        if end < text_length:
            # Find the next period or newline after chunk_size
            next_period = text.find('.', end)
            next_newline = text.find('\n', end)
            
            if next_period != -1 and (next_newline == -1 or next_period < next_newline):
                end = next_period + 1
            elif next_newline != -1:
                end = next_newline + 1
            else:
                # If no period or newline found, find the last space
                while end > start and text[end] != ' ':
                    end -= 1
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start = end - overlap
        
        # Ensure we don't get stuck
        if start >= end:
            break
    
    return chunks

def run_mistral(user_message, model="mistral-large-latest", max_retries=3):
    """Generate answer using Mistral with retry logic"""
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "user", "content": user_message}
            ]
            chat_response = client.chat.complete(
                model=model,
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise e

def answer_question(file, question):
    """Main function to process file and answer questions"""
    if file is None:
        return "Please upload a PDF or text file."
    
    try:
        # Extract text from file
        text = extract_text_from_file(file)
        
        if not text.strip():
            return "No text could be extracted from the file. Please ensure it contains readable text."
        
        # Create chunks
        chunks = create_chunks(text)
        
        if not chunks:
            return "No valid text chunks could be created from the file."
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Generate embeddings
        text_embeddings = process_chunks_with_rate_limit(chunks)
        
        # Create FAISS index
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)
        
        # Generate embedding for the question
        question_embeddings = np.array([get_text_embedding(question)])
        
        # Search for relevant chunks
        D, I = index.search(question_embeddings, k=2)
        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
        
        # Create prompt
        context = "\n".join(retrieved_chunks)
        prompt = f"""
        Context information is below.
        ---------------------
        {context}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {question}
        Answer:
        """
        
        return run_mistral(prompt)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.File(label="Upload File", file_types=[".pdf", ".txt"]),
        gr.Textbox(label="Ask a Question", placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Document Interaction Assistant with RAG and Mistral API",
    description="Upload a PDF or text file and ask questions about its content. The system will analyze the document and provide answers based on the content."
)

# Launch the Gradio app
interface.launch()