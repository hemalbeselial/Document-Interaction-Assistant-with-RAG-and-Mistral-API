# Document Interaction Assistant with RAG and Mistral API

# Description

This project implements a **Retrieval-Augmented Generation (RAG)** system using the Mistral API. It allows users to upload **PDF** and **TXT** documents, ask questions based on the content, and receive accurate answers. The user interface is built with **Gradio** and hosted on a local web server.

## Features

- **Document Upload**: Easily upload PDF and TXT files for processing.
- **Question-Answering**: Ask questions about the uploaded documents and receive content-driven responses.
- **Mistral API Integration**: Leverages Mistral's powerful API for efficient and accurate retrieval and generation of answers.
- **User-Friendly Interface**: A simple and intuitive interface built with Gradio for seamless interaction.

## Requirements

- **Python 3.12.2**
- **Gradio**
- **Mistral API Key**:
  - Get your API key from the official Mistral website.
  - Open `app.py` and paste the key into the `api_key` variable.
- Additional Python libraries (listed in `requirements.txt`).

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Start the application

python app.py

## Project Structure

Document-Interaction-Assistant-with-RAG-and-Mistral-API/

├── app.py                 # Main application file
|
├── requirements.txt       # List of dependencies
|
├── README.md              # Project documentation

## Example Workflow

1. Upload a PDF containing a user manual or reference document.
2. Ask questions such as:
      "What are the main features of the product?"
      "How do I troubleshoot common issues?"
3. Receive precise answers generated from the document content.

Feel free to reach out with any questions or feedback. Happy coding!
