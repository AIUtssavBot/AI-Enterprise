# app.py
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from rag.retrieval import rag_retriever
from utils.pdf_processor import process_pdf

# load_dotenv()
app = Flask(__name__)

load_dotenv()  # Load variables from .env file

print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))  # Debugging check

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set!")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    response = rag_retriever.retrieve_answer(user_input)
    return jsonify({"answer": response['answer']})

@app.route('/process-pdf', methods=['POST'])
def process_pdf_route():
    pdf_bytes = request.data
    documents = process_pdf(pdf_bytes)
    return jsonify({"message": "PDF processed successfully", "num_pages": len(documents)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
